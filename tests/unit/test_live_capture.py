"""Exercises the /ws/interview browser-capture endpoint via FastAPI's TestClient — a
real ASGI WebSocket round trip, in-process, no network. The AssemblyAI connection is
stubbed via dependency override + a scripted fake transport (same technique
test_web_backend.py uses for /chat with httpx.MockTransport).

SCOPE NOTE: this file proves the WEBSOCKET FRAMING AND RELAY layer — handshake
validation, which internal event becomes which outgoing message kind, and graceful
shutdown. It deliberately does NOT attempt to drive a full mic-audio-in -> turn-
evaluated-out round trip through this endpoint: that path crosses two independent
async hops (the browser's binary frames landing in SyncBus via `_audio_pump`'s
asyncio.to_thread hand-off, and the fake transport's scripted event replay) with no
shared clock, so forcing deterministic ordering between them would mean adding
sleep-based synchronization purely to satisfy a test — the flaky-test anti-pattern this
project's own test suite otherwise avoids. Turn finalization itself (that a
speech_started/stopped/transcript.user sequence produces a correct TurnEvidence) is
already covered deterministically, with no such race, by
tests/unit/test_live_interview_runner.py calling `_finalize_turn` directly.
"""
import base64
import json
import queue
import threading

import pytest
from fastapi.testclient import TestClient

import backend as backend_module
from voicestress.application.live_interview_runner import SAMPLE_RATE_HZ
from voicestress.config import AssemblyAISettings
from voicestress.infrastructure.models.keras_arousal_classifier import KerasArousalClassifier
from voicestress.infrastructure.models.resnet_light import build_resnet_light


class FakeAssemblyAITransport:
    """Scripted in-memory stand-in for a real AssemblyAI connection — same shape as
    test_voice_agent_client.py's FakeTransport, plus the async-context-manager methods
    `LiveInterviewRunner.run()`'s `async with await connect() as transport:` needs."""

    def __init__(self, incoming: list[dict]):
        self.sent: list[dict] = []
        self._incoming = list(incoming)

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        if not self._incoming:
            # Ends run()'s event loop via its catch-all `except Exception` branch —
            # the same behaviour a real dropped connection produces.
            raise ConnectionError("FakeAssemblyAITransport: scripted events exhausted")
        return json.dumps(self._incoming.pop(0))

    async def __aenter__(self) -> "FakeAssemblyAITransport":
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None


def _connect_with(incoming: list[dict]):
    async def connect():
        return FakeAssemblyAITransport(incoming)

    return connect


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    """`backend.py` and `live_interview_runner.py` each derive their own `SESSIONS_DIR`
    from their own file location — both land on the real
    `<repo>/artifacts/sessions/` in production, but they're two separate module-level
    constants, so a test must patch both or the runner's `_save_session()` (unaffected
    by patching only `backend_module.SESSIONS_DIR`) writes straight into the real
    repo directory. Caught here 2026-09-05: an earlier version of this fixture
    patched only the first and left six real files behind in the actual project."""
    import voicestress.application.live_interview_runner as runner_module

    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(backend_module, "SESSIONS_DIR", d)
    monkeypatch.setattr(runner_module, "SESSIONS_DIR", d)
    return d


@pytest.fixture
def model_path(tmp_path):
    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    path = tmp_path / "model.keras"
    model.save(str(path))
    return path


@pytest.fixture
def client(sessions_dir, model_path, monkeypatch):
    monkeypatch.setattr(backend_module, "MODEL_PATH", model_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "fake-key-not-sent-anywhere")
    # This whole file is specifically about live capture (ADR-041 onward) — enabled
    # here by default (ADR-049's flag defaults False for real deployments) so every
    # existing test keeps exercising the actual session flow; the handful of tests
    # for the disabled/default state override it back explicitly.
    monkeypatch.setattr(backend_module, "LIVE_CAPTURE_ENABLED", True)
    return TestClient(backend_module.app)


def _override_connect(fake_connect) -> None:
    backend_module.app.dependency_overrides[backend_module.get_voice_agent_connect] = (
        lambda: fake_connect
    )


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    backend_module.app.dependency_overrides.clear()


def _receive_with_timeout(ws, timeout_s: float = 5.0):
    """Starlette's WebSocketTestSession.receive() blocks on an internal stream with NO
    built-in timeout — if a mutated/broken server sends fewer frames than a test
    expects, `.receive()` hangs forever rather than raising. A hanging test is worse
    than a failing one: it doesn't just fail, it blocks the whole run. Confirmed the
    hard way, twice, while mutation-testing this file's own guards (ADR-039):
    mutating away the agent-audio binary-frame branch made
    `test_agent_reply_audio_is_relayed_as_a_raw_binary_frame` hang instead of fail —
    and the FIRST fix attempt (`ThreadPoolExecutor` as a context manager) hung too,
    because its `__exit__` calls `shutdown(wait=True)`, which blocks on the same
    abandoned worker thread still stuck inside the real `.receive()` call. A daemon
    `threading.Thread` that nothing ever joins, reporting through a `queue.Queue`,
    does not have that failure mode: an abandoned worker is simply leaked, and daemon
    threads don't block interpreter or test-process exit.
    """
    result: "queue.Queue" = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: result.put(ws.receive()), daemon=True).start()
    try:
        return result.get(timeout=timeout_s)
    except queue.Empty:
        pytest.fail(f"ws.receive() did not return within {timeout_s}s — likely hung")


def _drain_until_json(ws, predicate, max_tries: int = 25) -> dict:
    """Reads frames until a JSON text frame satisfies `predicate`. Necessary because
    runner._status() writes a free-text status log for EVERY event (mirroring the
    terminal script's print() output) in addition to any structured message a
    particular event also produces — several status messages can share
    `type == "status"`, so a test looking for one specific status needs to filter on
    content, not just type, and skip whichever others happen to arrive first."""
    for _ in range(max_tries):
        message = _receive_with_timeout(ws)
        if message.get("text") is not None:
            parsed = json.loads(message["text"])
            if predicate(parsed):
                return parsed
        # binary frames are skipped over — not what this helper is looking for
    pytest.fail(f"no matching message received within {max_tries} frames")


def _drain_until_json_type(ws, wanted_type: str, max_tries: int = 25) -> dict:
    return _drain_until_json(ws, lambda m: m.get("type") == wanted_type, max_tries)


def _drain_until_bytes(ws, max_tries: int = 25) -> bytes:
    for _ in range(max_tries):
        message = _receive_with_timeout(ws)
        if message.get("bytes") is not None:
            return message["bytes"]
    pytest.fail(f"never received a binary frame within {max_tries} frames")


def test_rejects_a_capture_rate_that_is_not_24khz(client):
    _override_connect(_connect_with([]))
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": 16_000})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "16000" in msg["message"] or "16_000" in msg["message"]
        assert str(SAMPLE_RATE_HZ) in msg["message"]


def test_rejects_a_missing_sample_rate_field(client):
    _override_connect(_connect_with([]))
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_accepted_handshake_gets_a_session_started_message(client):
    _override_connect(_connect_with([{"type": "session.updated", "config": {"id": "sess_1"}}]))
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": SAMPLE_RATE_HZ})
        msg = ws.receive_json()
        assert msg["type"] == "session_started"
        assert msg["session_id"].startswith("s_")


def test_agent_text_is_relayed_as_its_own_message_kind(client):
    """No candidate audio at all in this script — proves the relay in isolation,
    without touching the audio-pump timing this file's docstring excludes."""
    _override_connect(
        _connect_with(
            [
                {"type": "transcript.agent", "text": "Tell me about yourself."},
            ]
        )
    )
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": SAMPLE_RATE_HZ})
        ws.receive_json()  # session_started

        msg = _drain_until_json_type(ws, "agent_text")
        assert msg == {"type": "agent_text", "text": "Tell me about yourself."}


def test_agent_reply_audio_is_relayed_as_a_raw_binary_frame(client):
    """ADR-040: the agent's synthesized reply, decoded from AssemblyAI's base64 JSON
    envelope and re-sent to the browser as a plain binary frame — capture.js plays it
    straight through Web Audio with no further decoding."""
    raw_pcm = bytes([1, 2, 3, 4, 250, 251])
    _override_connect(
        _connect_with(
            [{"type": "reply.audio", "audio": base64.b64encode(raw_pcm).decode()}]
        )
    )
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": SAMPLE_RATE_HZ})
        ws.receive_json()  # session_started

        frame = _drain_until_bytes(ws)
        assert frame == raw_pcm


def test_malformed_binary_frame_is_dropped_with_a_status_not_a_crash(client):
    """An odd byte count cannot be a whole number of int16 samples (ADR-027: every
    branch here must say what happened rather than silently drop or crash)."""
    _override_connect(_connect_with([]))
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": SAMPLE_RATE_HZ})
        ws.receive_json()  # session_started

        ws.send_bytes(b"\x01\x02\x03")  # 3 bytes: not divisible by 2

        _drain_until_json(
            ws, lambda m: m.get("type") == "status" and "dropped" in m.get("message", "")
        )


def test_end_session_control_message_saves_and_confirms(client, sessions_dir):
    _override_connect(_connect_with([]))
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": SAMPLE_RATE_HZ})
        started = ws.receive_json()
        session_id = started["session_id"]

        ws.send_json({"type": "end_session"})

        # Drain status/turn messages until the terminal one arrives, rather than
        # assuming it's the very next frame — the runner's own shutdown status
        # ("Saved session evidence to ...") is enqueued ahead of it.
        msg = _drain_until_json_type(ws, "session_saved")
        assert msg["session_id"] == session_id

    assert (sessions_dir / f"{session_id}.json").exists()


def test_reports_error_not_session_saved_when_the_file_was_never_written(client, sessions_dir, monkeypatch):
    """ADR-042: a cancellation that lands before run()'s own save logic runs used to
    still report "session_saved" — a claim of success with nothing on disk to back it.
    Forces the save itself to fail (SESSIONS_DIR points at a plain file, so `.mkdir()`
    inside `_save_session()` raises) and checks the honest outcome is reported instead."""
    import voicestress.application.live_interview_runner as runner_module

    not_a_directory = sessions_dir.parent / "not_a_directory"
    not_a_directory.write_text("occupying this path so mkdir() on it fails")
    monkeypatch.setattr(runner_module, "SESSIONS_DIR", not_a_directory)

    _override_connect(_connect_with([]))
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": SAMPLE_RATE_HZ})
        ws.receive_json()  # session_started

        ws.send_json({"type": "end_session"})

        msg = _drain_until_json(
            ws, lambda m: m.get("type") in ("session_saved", "error") and "runner task" not in m.get("message", "")
        )
        assert msg["type"] == "error", f"claimed success with nothing written: {msg}"
        assert "no file was written" in msg["message"]


def test_missing_api_key_is_rejected_before_any_session_starts(client, monkeypatch):
    monkeypatch.setattr(backend_module, "LIVE_CAPTURE_ENABLED", True)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with client.websocket_connect("/ws/interview") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "not configured" in msg["message"]


def test_live_capture_disabled_by_default_rejects_immediately(client, monkeypatch):
    """ADR-049: a deployment can enable the Analyst Agent chat (a real key configured)
    while keeping live capture off — the flag defaults to disabled, refusing even
    before checking whether a key exists, so setting a key for /chat can never
    accidentally also switch this on. (The `client` fixture enables the flag for the
    rest of this file, since it's specifically about live capture — turned back off
    here to test the real default.)"""
    monkeypatch.setattr(backend_module, "LIVE_CAPTURE_ENABLED", False)
    with client.websocket_connect("/ws/interview") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "disabled" in msg["message"]


def test_live_capture_stays_disabled_even_with_a_real_key_configured(client, monkeypatch):
    """The critical case: setting ASSEMBLYAI_API_KEY (to power /chat publicly) must
    NOT also enable live capture as a side effect. The flag is checked independently
    of, and before, the key."""
    monkeypatch.setattr(backend_module, "LIVE_CAPTURE_ENABLED", False)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "a-real-looking-key")
    with client.websocket_connect("/ws/interview") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "disabled" in msg["message"]


@pytest.mark.parametrize("value", ["true", "True", "TRUE", " true "])
def test_env_flag_reads_true_case_and_whitespace_insensitively(monkeypatch, value):
    monkeypatch.setenv("VOICESTRESS_ENABLE_LIVE_CAPTURE", value)
    assert backend_module._env_flag("VOICESTRESS_ENABLE_LIVE_CAPTURE") is True


@pytest.mark.parametrize("value", ["false", "0", "yes", "", "nonsense"])
def test_env_flag_treats_anything_else_as_false(monkeypatch, value):
    monkeypatch.setenv("VOICESTRESS_ENABLE_LIVE_CAPTURE", value)
    assert backend_module._env_flag("VOICESTRESS_ENABLE_LIVE_CAPTURE") is False


def test_env_flag_uses_the_default_when_unset(monkeypatch):
    monkeypatch.delenv("VOICESTRESS_ENABLE_LIVE_CAPTURE", raising=False)
    assert backend_module._env_flag("VOICESTRESS_ENABLE_LIVE_CAPTURE", default=False) is False
    assert backend_module._env_flag("VOICESTRESS_ENABLE_LIVE_CAPTURE", default=True) is True


def test_unknown_control_message_is_ignored_with_a_status(client):
    _override_connect(_connect_with([]))
    with client.websocket_connect("/ws/interview") as ws:
        ws.send_json({"type": "handshake", "sample_rate_hz": SAMPLE_RATE_HZ})
        ws.receive_json()  # session_started

        ws.send_json({"type": "pause"})  # not a recognised control message

        _drain_until_json(
            ws,
            lambda m: m.get("type") == "status"
            and "unknown control message" in m.get("message", ""),
        )
