"""Bridges: audio source -> SyncBus -> EvidenceService -> InterviewSession, while a
real VoiceAgentSession drives the actual conversation. Moved here from
scripts/run_interview.py (2026-09-05, ADR-041) so a browser-based capture surface
(web/backend.py's `/ws/interview`) can share this orchestration instead of duplicating
it — this file has ~15 documented bug fixes baked into its comments (ADR.md Appendix A),
and copy-pasting it into a second composition root would have meant re-discovering every
one of them independently.

The Interview Agent itself (`voice_session`) never receives anything derived from
`evidence_service` — see ARCHITECTURE.md §2. This class only *listens* to the agent's
turn-boundary events and reply events; it never calls back into the Voice Agent session
with analysis results. `on_turn_finalized` exists for a human operator's screen (terminal
print, or a browser dashboard) — the same audience the original terminal script's print()
statements already had. It is not, and must never become, a path back into the Interview
Agent's own context; only `VoiceAgentSession`'s constructor arguments can do that, and it
has no argument shaped to accept one (tests/unit/test_voice_agent_client.py guards this).
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import queue
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import numpy as np
import websockets.exceptions

from voicestress.application.asr_signals import mean_word_confidence, onset_latency_ms
from voicestress.application.baseline_service import BaselineCalibrator
from voicestress.application.evidence_service import EvidenceService
from voicestress.application.sync_bus import PendingTurn, SyncBus
from voicestress.application.transcript_pairing import TranscriptPairer
from voicestress.config import AssemblyAISettings
from voicestress.domain.entities import InterviewSession, TurnEvidence
from voicestress.domain.value_objects import SpeakerId
from voicestress.infrastructure.agents.voice_agent_client import VoiceAgentSession
from voicestress.infrastructure.agents.websocket_transport import WebsocketTransport
from voicestress.infrastructure.audio.prosody import PraatProsodyExtractor
from voicestress.infrastructure.audio.spectrogram import NarrowbandSpectrogramExtractor
from voicestress.infrastructure.models.gradcam import GradCAMExplainer
from voicestress.infrastructure.models.keras_arousal_classifier import KerasArousalClassifier

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSIONS_DIR = ROOT / "artifacts" / "sessions"
ARTIFACT_DIR = ROOT / "artifacts" / "turn_artifacts"
PROMPT_PATH = ROOT / "agents" / "prompts" / "interviewer.md"

SAMPLE_RATE_HZ = 24_000  # Voice Agent API's rate — ARCHITECTURE.md §5
CHUNK_MS = 100
CHUNK_SAMPLES = SAMPLE_RATE_HZ * CHUNK_MS // 1000
NUM_BASELINE_TURNS = 3  # ARCHITECTURE.md §7
MIN_TURN_DURATION_MS = 300  # ARCHITECTURE.md §7b — filters spurious VAD blips (a real
# 100ms mic-click/noise "turn" reached the pipeline on the first live run, 2026-09-05)

StatusHandler = Callable[[str], None]
TurnHandler = Callable[[TurnEvidence], Awaitable[None] | None]
TextHandler = Callable[[str], Awaitable[None] | None]
AudioHandler = Callable[[bytes], Awaitable[None] | None]

# ADR-044: the ONE tool the Interview Agent is ever configured with. Confirmed live
# 2026-09-06 that this account's Voice Agent API supports tool-calling at all (a
# separate entitlement path from the LLM Gateway's chat-completions tool-calling used
# by the Analyst Agent, which this same account cannot reach on any of 34 tested
# models — see ADR-024's update). This tool reports a candidate's own statement that
# something is technically wrong (echo, can't hear, audio cutting out) — pure session
# metadata for whoever reviews the recording later. It has no `turn_id`, no score, no
# path anywhere near `evidence_service`; ADR-002's guarantee is a fact about what data
# CAN reach this agent, and this tool schema simply never mentions anything that would.
FLAG_TECHNICAL_ISSUE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "flag_technical_issue",
    "description": (
        "Call this if the candidate reports an audio or connection problem — echo, "
        "can't hear you, choppy or cutting-out audio, etc. This only logs a data-"
        "quality note for whoever reviews the recording; it does not change anything "
        "about how the interview proceeds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "issue": {
                "type": "string",
                "description": "Brief description of the problem, in the candidate's own words if possible.",
            }
        },
        "required": ["issue"],
    },
}


class AudioSource(Protocol):
    """What LiveInterviewRunner needs from wherever the candidate's audio actually
    comes from. `start` must return promptly — for a real microphone that means
    starting a background stream/thread that calls `feed` per chunk; for a browser
    capture socket it means simply recording the callback, since chunks arrive by
    whatever is already reading the socket. `feed` receives mono int16 PCM at
    SAMPLE_RATE_HZ; the source is responsible for getting audio into that shape
    before calling it — this class asserts nothing about where the audio originated.
    """

    def start(self, feed: Callable[[np.ndarray], None]) -> None: ...
    def stop(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class TranscriptPayload:
    """What TranscriptPairer now carries through its FIFO queue for this script —
    text plus the word-level confidence/timestamp data AssemblyAI's streaming API
    returns (confirmed 2026-09-06), which was previously discarded (only `.text` was
    ever read from the transcript.user event). Default-constructible with empty
    values so TranscriptPairer's overflow/flush paths (a turn whose transcript never
    arrives) still produce a valid, empty payload rather than needing a None-check
    at every call site.
    """

    text: str = ""
    words: tuple[dict, ...] = ()


class LiveInterviewRunner:
    """Bridges: audio source -> SyncBus -> EvidenceService -> InterviewSession, while a
    real VoiceAgentSession drives the actual conversation. The Interview Agent itself
    (`voice_session`) never receives anything derived from `evidence_service` — see
    ARCHITECTURE.md §2. This class only *listens* to the agent's turn-boundary events;
    it never calls back into it with analysis results.
    """

    def __init__(
        self,
        settings: AssemblyAISettings,
        model_path: Path,
        on_status: StatusHandler | None = None,
        on_turn_finalized: TurnHandler | None = None,
        # ADDED 2026-09-05 (ADR-040): the agent's own reply — text and (best-effort)
        # audio — surfaced to whoever is running this session. Neither of these
        # touches evidence_service or voice_session's inbound side; they only observe
        # what the Interview Agent already sent.
        on_agent_reply_text: TextHandler | None = None,
        on_agent_reply_audio: AudioHandler | None = None,
    ):
        self.settings = settings
        self.session_id = f"s_{int(time.time())}"
        self.speaker = SpeakerId(corpus="interview", raw_id="candidate")
        self.interview_session = InterviewSession(session_id=self.session_id, speaker=self.speaker)
        self.sync_bus = SyncBus(
            sample_rate_hz=SAMPLE_RATE_HZ, min_turn_duration_ms=MIN_TURN_DURATION_MS
        )
        self.pairer: TranscriptPairer[PendingTurn, TranscriptPayload] = TranscriptPairer(
            default_payload_factory=TranscriptPayload
        )
        self.calibrator = BaselineCalibrator(speaker=self.speaker)
        self.turn_count = 0
        self.baseline_ready = False
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._on_status = on_status
        self._on_turn_finalized = on_turn_finalized
        self._on_agent_reply_text = on_agent_reply_text
        self._on_agent_reply_audio = on_agent_reply_audio
        # Set inside run(), once the live VoiceAgentSession exists — needed so
        # _on_tool_call can send a tool.result back to close the loop the Voice Agent
        # API expects. None until then (and again after the session ends); every use
        # site checks for that.
        self._voice_session: VoiceAgentSession | None = None

        classifier = KerasArousalClassifier.from_checkpoint(model_path, version="v1-warm_started")
        self.evidence_service = EvidenceService(
            spectrogram_extractor=NarrowbandSpectrogramExtractor(),
            prosody_extractor=PraatProsodyExtractor(),
            classifier=classifier,
            explainer=GradCAMExplainer(model=classifier.model),
            artifact_dir=ARTIFACT_DIR,
        )

    def _status(self, message: str) -> None:
        """Every diagnostic in this class goes through here — printed for the terminal
        script's operator exactly as before, and forwarded to `on_status` for a web
        caller's live status feed. Kept as one call site so no future edit can add a
        print() with no matching hook, or vice versa.

        TRUNCATED here (ADR-043) after a real live session froze the browser tab: the
        `_on_reply_audio` ASSUMPTION about the payload's key name was wrong (confirmed
        live, 2026-09-06 — see that method's comment), so EVERY reply.audio event hit
        its "log the raw event" fallback, each dumping a multi-kilobyte base64 blob as
        one status line. capture.js appended each as a full DOM text node with no cap;
        after a few exchanges' worth of chunks, the tab stopped responding. Truncating
        here is the general fix — it protects every current and future call site, not
        only this one event type, from a single unexpectedly large payload doing the
        same thing again.
        """
        MAX_STATUS_CHARS = 500
        if len(message) > MAX_STATUS_CHARS:
            message = f"{message[:MAX_STATUS_CHARS]}... [truncated, {len(message)} chars total]"
        print(message)
        if self._on_status is not None:
            self._on_status(message)

    def feed_audio_chunk(self, chunk_i16: np.ndarray) -> None:
        """Public entry point for any AudioSource to hand over one chunk of mono int16
        PCM at SAMPLE_RATE_HZ. Replaces the old sounddevice-specific `_mic_callback`,
        which wrote to this same queue directly."""
        self._audio_queue.put(chunk_i16)

    async def _on_transcript(self, event: dict) -> None:
        # CONFIRMED correct 2026-09-05: "text" is the right field — real transcripts
        # ("Hello?", "Can you hear me?", ...) came through it on the first live run.
        # What was WRONG (before TranscriptPairer) is now fixed: transcript.user
        # (final) for a turn can arrive well after that turn's input.speech.stopped —
        # observed directly as a 300ms-audio turn (t0007) logged with an 8-word
        # transcript that belonged to an earlier turn. TranscriptPairer fixes the
        # pairing; see its module docstring.
        #
        # ADDED 2026-09-06: "words" — confirmed available in AssemblyAI's streaming
        # docs (per-word confidence 0-1 and start/end ms timestamps) — was always
        # present on this event and previously discarded entirely. Now carried through
        # via TranscriptPayload so _finalize_turn can derive asr_mean_confidence and a
        # real onset_latency_ms instead of leaving that field permanently None.
        payload = TranscriptPayload(
            text=event.get("text", ""), words=tuple(event.get("words") or ())
        )
        paired = self.pairer.submit_transcript(payload)
        if paired is not None:
            pending, transcript_payload = paired
            await self._finalize_turn(pending, transcript_payload)

    async def _on_speech_started(self, event: dict) -> None:
        self.sync_bus.on_speech_started()

    async def _on_speech_stopped(self, event: dict) -> None:
        pending = self.sync_bus.on_speech_stopped()
        if pending is None:
            self._status(
                f"[discarded] speech segment shorter than {MIN_TURN_DURATION_MS}ms "
                f"(likely VAD noise, not real speech) — "
                f"total discarded so far: {self.sync_bus.turns_discarded_too_short}"
            )
            return

        overflow = self.pairer.submit_turn(pending)
        if overflow is not None:
            # An older turn's transcript never arrived (e.g. wordless audio) and the
            # pending queue hit its ceiling — finalize that older turn now with an
            # empty transcript rather than blocking every pairing after it.
            stale_pending, empty_payload = overflow
            await self._finalize_turn(stale_pending, empty_payload)
        # else: queued, awaiting its transcript — finalized later from _on_transcript
        # (or by flush_all() at session end if it never arrives).

    async def _on_agent_transcript(self, event: dict) -> None:
        # ADDED 2026-09-05 (ADR-040): "transcript.agent" was always in the dispatch
        # table (voice_agent_client.py) but no composition root ever passed a handler
        # for it — the agent's spoken reply, transcribed to text, was silently
        # dropped by every prior live run. See ADR-040.
        text = event.get("text", "")
        if not text:
            return
        self._status(f"[agent] {text}")
        if self._on_agent_reply_text is not None:
            result = self._on_agent_reply_text(text)
            if result is not None:
                await result

    async def _on_reply_started(self, event: dict) -> None:
        self._status("[agent replying...]")

    # CONFIRMED 2026-09-06 against AssemblyAI's own published docs (tool-calling guide
    # + the 5-minute walkthrough's example handler, `event['data']`): the key is
    # "data", not the originally guessed "audio" (ADR-043's browser-freeze bug), nor
    # any of the other candidates tried in between. "data" is listed first now that
    # it's a known fact, not a guess; the rest stay as a safety net; the ASSUMPTION
    # this comment used to carry is retired — see ADR-045.
    _REPLY_AUDIO_CANDIDATE_KEYS = ("data", "audio", "delta", "chunk", "pcm", "audio_delta")

    async def _on_reply_audio(self, event: dict) -> None:
        raw = None
        used_key = None
        for key in self._REPLY_AUDIO_CANDIDATE_KEYS:
            value = event.get(key)
            if isinstance(value, str) and value:
                raw, used_key = value, key
                break

        if raw is None:
            # Deliberately NOT logging event values here — only keys and each value's
            # type/length. A wrong-key guess must never again turn into a multi-
            # kilobyte base64 blob written to a human-facing log (ADR-043).
            shape = {k: f"{type(v).__name__}(len={len(v) if hasattr(v, '__len__') else '?'})"
                     for k, v in event.items()}
            self._status(
                f"[reply.audio] none of {self._REPLY_AUDIO_CANDIDATE_KEYS} matched a "
                f"non-empty string field; event shape was {shape}"
            )
            return
        try:
            pcm_bytes = base64.b64decode(raw)
        except Exception as e:  # noqa: BLE001 — printed per ADR-027, never swallowed
            self._status(
                f"[reply.audio] failed to decode payload under key {used_key!r} "
                f"(length {len(raw)}): {type(e).__name__}: {e}"
            )
            return
        if self._on_agent_reply_audio is not None:
            result = self._on_agent_reply_audio(pcm_bytes)
            if result is not None:
                await result

    async def _on_reply_done(self, event: dict) -> None:
        self._status("[agent reply complete]")

    async def _on_tool_call(self, event: dict) -> None:
        # ADR-044: the ONLY tool.call this class will ever receive is the one tool it
        # configures the Interview Agent with (see FLAG_TECHNICAL_ISSUE_SCHEMA). A
        # future edit adding a second tool must extend this dispatch deliberately, not
        # silently accept whatever the server sends — hence the explicit else-branch
        # rather than assuming `name` is always the one we expect.
        name = event.get("name")
        call_id = event.get("call_id")
        if name != "flag_technical_issue":
            self._status(f"[tool.call] unexpected tool name {name!r} — ignoring: {event}")
            return

        issue = (event.get("arguments") or {}).get("issue", "")
        self.interview_session.technical_flags.append(issue)
        self._status(f"[technical issue flagged by candidate] {issue}")

        if self._voice_session is not None and call_id is not None:
            await self._voice_session.send_tool_result(call_id, {"logged": True})

    async def _finalize_turn(self, pending: PendingTurn, transcript_payload: TranscriptPayload) -> None:
        self.turn_count += 1
        is_baseline = self.turn_count <= NUM_BASELINE_TURNS
        transcript = transcript_payload.text

        profile = self.calibrator.build_profile() if self.baseline_ready else None
        # OFFLOADED TO A THREAD 2026-09-06 (ADR-028): build_turn_evidence runs the CNN
        # forward pass, Grad-CAM (a second forward+backward pass), and Praat pitch
        # extraction — all synchronous, all CPU-bound, together worth well over a
        # second per turn. Calling this directly (as the original version did) blocks
        # the ENTIRE asyncio event loop for that duration: no incoming Voice Agent
        # events get processed and no outgoing audio gets sent while it runs. Combined
        # with SyncBus's now-fixed O(n^2) buffer bug, this compounding delay was the
        # real mechanism behind turns 10-14 going degenerate on a real 14-turn session
        # — not model instability, a real-time backlog. asyncio.to_thread keeps this
        # heavy work off the loop so event/audio processing continues concurrently.
        evidence = await asyncio.to_thread(
            self.evidence_service.build_turn_evidence,
            session_id=self.session_id,
            turn_id=pending.turn_id,
            t_start_ms=pending.t_start_ms,
            t_end_ms=pending.t_end_ms,
            is_baseline_turn=is_baseline,
            transcript=transcript,
            samples=pending.samples,
            sample_rate_hz=SAMPLE_RATE_HZ,
            baseline_profile=profile,
        )

        # ADDED 2026-09-06: ASR-derived signals from AssemblyAI's own word-level data,
        # independent of the local Praat/CNN pipeline — see application/asr_signals.py.
        # ProsodyFeatures is frozen, so patching in these two fields (populated from
        # event-level context build_turn_evidence never sees, same as onset_latency_ms
        # was always designed to be) means reconstructing it via dataclasses.replace.
        asr_confidence = mean_word_confidence(transcript_payload.words)
        onset_latency = onset_latency_ms(pending.t_start_ms, transcript_payload.words)
        if asr_confidence is not None or onset_latency is not None:
            evidence.prosody_raw = dataclasses.replace(
                evidence.prosody_raw,
                asr_mean_confidence=asr_confidence,
                onset_latency_ms=onset_latency,
            )

        self.interview_session.add_turn(evidence)

        if is_baseline:
            self.calibrator.add_turn(evidence.prosody_raw)
            if self.turn_count == NUM_BASELINE_TURNS:
                self.baseline_ready = True
                self._status(f"[baseline] calibrated from {NUM_BASELINE_TURNS} opening turns")

        label = "baseline" if is_baseline else "scored"
        self._status(
            f"[{pending.turn_id}] ({label}) arousal={evidence.arousal.probability:.2f} "
            f"transcript={transcript[:60]!r}"
        )
        if self._on_turn_finalized is not None:
            result = self._on_turn_finalized(evidence)
            if result is not None:
                await result

    async def _audio_pump(self, voice_session: VoiceAgentSession) -> None:
        while True:
            chunk_i16 = await asyncio.to_thread(self._audio_queue.get)
            self.sync_bus.feed_audio(chunk_i16.astype(np.float32) / 32768.0)
            await voice_session.send_audio_chunk(chunk_i16.tobytes())

    def _save_session(self) -> Path:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SESSIONS_DIR / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "turns": [t.to_evidence_dict() for t in self.interview_session.turns],
            "technical_flags": self.interview_session.technical_flags,
        }
        out_path.write_text(json.dumps(data, indent=2))
        return out_path

    async def run(
        self,
        audio_source: AudioSource,
        connect: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        """Runs one full interview session against the real Voice Agent API, sourcing
        the candidate's audio from `audio_source` — a real microphone
        (`infrastructure/audio/mic_source.py`, the terminal path) or a browser capture
        socket (`web/live_capture.py`, the web path). Everything below this line is
        transport-agnostic; it was audio-source-specific only in exactly the two spots
        marked `audio_source.start`/`.stop()` below (moved out of a `sounddevice`
        `InputStream` call and its teardown, ADR-041 — every other line, including
        every exception branch, is unchanged from the original terminal-only version
        and carries the live-run history that produced it).

        `connect` overrides how the real Voice Agent connection is opened — production
        callers leave it as the default (a real `WebsocketTransport.connect(...)`,
        which genuinely cannot be exercised without a live ASSEMBLYAI_API_KEY, same
        boundary `websocket_transport.py`'s own docstring names). Tests inject a
        connect function returning a fake, async-context-manager-shaped transport, so
        this method's actual orchestration — event dispatch, turn finalization,
        graceful shutdown — is exercisable with no network at all.
        """
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        connect = connect or (
            lambda: WebsocketTransport.connect(self.settings.voice_agent_ws_url, self.settings.api_key)
        )

        async with await connect() as transport:
            voice_session = VoiceAgentSession(
                transport=transport,
                on_transcript_final=self._on_transcript,
                on_speech_started=self._on_speech_started,
                on_speech_stopped=self._on_speech_stopped,
                on_agent_transcript=self._on_agent_transcript,
                on_reply_started=self._on_reply_started,
                on_reply_audio=self._on_reply_audio,
                on_reply_done=self._on_reply_done,
                on_tool_call=self._on_tool_call,
                on_session_ready=lambda e: self._status(f"[session ready] {e.get('session_id')}"),
                # CONFIRMED 2026-09-05 (scripts/check_live_connection.py): this
                # account's connection emits session.updated, not session.ready — wire
                # both so whichever fires is logged.
                on_session_updated=lambda e: self._status(
                    f"[session updated] {(e.get('config') or {}).get('id')}"
                ),
                # ADDED after a real disconnect (2026-09-05) that ended the session
                # after a single turn with zero explanation — see
                # voice_agent_client.py's dispatch comment. Whatever this prints next
                # time is the actual server-given reason.
                on_session_ended=lambda e: self._status(f"[session ended by server] {e}"),
                on_error=lambda e: self._status(f"[session error] {e}"),
            )
            self._voice_session = voice_session

            # `pump_task` is created inside the try below; declared here so the
            # `finally` can check it without a NameError if cancellation lands before
            # that line ever runs (see the ADR-041/ADR-042 comment on that `finally`).
            pump_task: asyncio.Task | None = None
            try:
                # Exactly one tool, confirmed live 2026-09-06 to actually work on this
                # account's Voice Agent API (ADR-044) — data-quality metadata only.
                # The Interview Agent still stays blind to arousal analysis by
                # construction (ARCHITECTURE.md §2): FLAG_TECHNICAL_ISSUE_SCHEMA has
                # nothing acoustic to call, and _on_tool_call's dispatch rejects any
                # tool name it doesn't recognize rather than acting on it blindly.
                await voice_session.configure(
                    tools=[FLAG_TECHNICAL_ISSUE_SCHEMA], system_prompt=prompt
                )

                audio_source.start(self.feed_audio_chunk)
                pump_task = asyncio.create_task(self._audio_pump(voice_session))

                self._status(f"Session {self.session_id} starting. Ctrl+C to end.")
                while True:
                    await voice_session.handle_next_event()
            except websockets.exceptions.ConnectionClosed as e:
                # ADDED after a real run (2026-09-05) that ended silently after one
                # turn: websockets.exceptions.ConnectionClosed is NOT a subclass of
                # Python's built-in ConnectionError (confirmed via its MRO), so the
                # previous `except (ConnectionError, ...)` never caught it — the
                # exception was propagating past `finally` as an unhandled crash with
                # no diagnostic printed before it. `.code`/`.reason` are the server's
                # actual stated cause (WebSocket close frame); print them instead of
                # guessing.
                self._status(
                    f"[connection closed] code={e.code} reason={e.reason!r} ({type(e).__name__})"
                )
            except KeyboardInterrupt:
                self._status("[ended] Ctrl+C")
            except asyncio.CancelledError:
                # NOT re-raised: matches the original terminal script's behaviour
                # exactly. A caller that does `task.cancel(); await task` (the web
                # endpoint's graceful-stop and disconnect paths, ADR-041) sees this
                # coroutine complete normally rather than needing to catch
                # CancelledError itself — the `finally` block below still runs and
                # still saves the session either way.
                self._status("[ended] task cancelled")
            except Exception as e:
                # ADDED 2026-09-06 after a THIRD silent-exit report: a session ended
                # after one turn with ZERO diagnostic output — neither the
                # ConnectionClosed branch above nor a session.ended event fired. The
                # previous version of this except clause was `except (ConnectionError,
                # KeyboardInterrupt, asyncio.CancelledError): pass` — a raw
                # ConnectionResetError/OSError (both builtin ConnectionError
                # subclasses, plausible on an abrupt TCP-level reset rather than a
                # clean WebSocket close handshake — not uncommon on Windows) would
                # have been swallowed by that `pass` with nothing printed. Rather than
                # add a fourth narrow special case, this is a catch-all: whatever
                # exception type reaches here, print its type and message before
                # falling through to cleanup. This is the actual fix — the third
                # occurrence of the same silence pattern (ADR.md Appendix A, bugs
                # 12/13) means the bug class is "an except clause that doesn't print",
                # not any one specific exception type.
                self._status(f"[session loop ended unexpectedly] {type(e).__name__}: {e}")
            finally:
                # ADR-042: this `finally` used to sit only around the `while True:
                # handle_next_event()` loop, one indent shallower than it is now — a
                # cancellation landing during `configure()` or before `pump_task` was
                # even created (confirmed live: a client that sends "end_session"
                # within milliseconds of "session_started") propagated straight out of
                # `run()` without ever reaching here. `web/live_capture.py` then
                # unconditionally reported `{"type": "session_saved"}` after awaiting
                # the cancelled task — a claim of success with no file ever written.
                # Wrapping the whole body, not just the event loop, guarantees flush +
                # save happen before ANY exit from this method, cancellation included.
                if pump_task is not None:
                    pump_task.cancel()
                audio_source.stop()
                self._voice_session = None  # transport is closing; nothing left to send a tool.result to

                # Any turns still queued in the pairer never got a matching transcript
                # (session ended before it arrived, or before max_pending overflowed
                # it) — flush them now with empty transcript rather than silently
                # dropping the last turn(s) of the conversation from the saved session.
                for pending, empty_payload in self.pairer.flush_all():
                    await self._finalize_turn(pending, empty_payload)

                out_path = self._save_session()
                self._status(f"Saved session evidence to {out_path}")
