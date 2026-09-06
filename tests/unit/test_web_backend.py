"""Exercises the dashboard backend with FastAPI's TestClient — real HTTP request/
response cycle, in-process, no network. The chat endpoint's LLM Gateway call is stubbed
via dependency override + httpx.MockTransport (same technique as
tests/unit/test_llm_gateway_client.py), so this covers real routing/wiring without
needing ASSEMBLYAI_API_KEY.
"""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import backend as backend_module
from voicestress.config import AssemblyAISettings


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(backend_module, "SESSIONS_DIR", d)
    return d


@pytest.fixture
def client(sessions_dir):
    return TestClient(backend_module.app)


def _write_session(sessions_dir, session_id: str, turns: list[dict], **extra) -> None:
    payload = {"session_id": session_id, "turns": turns, **extra}
    (sessions_dir / f"{session_id}.json").write_text(json.dumps(payload))


def _turn_dict(turn_id="t0001", score=0.7, baseline=False) -> dict:
    return {
        "session_id": "s1",
        "turn_id": turn_id,
        "t_start_ms": 0,
        "t_end_ms": 2000,
        "is_baseline_turn": baseline,
        "transcript": "Q: why? A: because.",
        "stress": {"score": score, "label": "high" if score >= 0.5 else "low", "model_version": "v1", "calibrated": True},
        "baseline_deviation": {"f0_mean_z": 1.2},
        "prosody_raw": {
            "f0_mean_hz": 180, "f0_std_hz": 20, "jitter_local_pct": 1, "shimmer_local_pct": 5,
            "hnr_db": 15, "speech_rate_syll_s": 4, "onset_latency_ms": None,
        },
        "xai": {"gradcam_png": None, "spectrogram_png": None, "top_contributors": []},
        "disclaimer": "Vocal stress signal relative to this speaker's own baseline. Not a determination of truthfulness.",
    }


def test_list_sessions_empty_when_no_sessions(client):
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_sessions_returns_summary(client, sessions_dir):
    _write_session(sessions_dir, "s1", [_turn_dict()], is_synthetic_demo=True, note="demo note")
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["session_id"] == "s1"
    assert body[0]["n_turns"] == 1
    assert body[0]["is_synthetic_demo"] is True


def test_get_session_returns_full_payload(client, sessions_dir):
    _write_session(sessions_dir, "s1", [_turn_dict()])
    resp = client.get("/api/sessions/s1")
    assert resp.status_code == 200
    assert resp.json()["turns"][0]["turn_id"] == "t0001"


def test_get_session_404_when_missing(client, sessions_dir):
    resp = client.get("/api/sessions/does_not_exist")
    assert resp.status_code == 404


def test_get_session_rejects_path_traversal(client, sessions_dir):
    resp = client.get("/api/sessions/..%2F..%2Fsecrets")
    assert resp.status_code in (400, 404)  # never 200, never leaks a file outside sessions_dir


def test_get_artifact_404_when_no_path_recorded(client, sessions_dir):
    _write_session(sessions_dir, "s1", [_turn_dict()])  # gradcam_png/spectrogram_png are None
    resp = client.get("/api/sessions/s1/artifact/t0001/gradcam")
    assert resp.status_code == 404


def test_get_artifact_serves_real_file(client, sessions_dir, tmp_path):
    img_path = tmp_path / "fake_gradcam.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)  # minimal PNG-ish header
    turn = _turn_dict()
    turn["xai"]["gradcam_png"] = str(img_path)
    _write_session(sessions_dir, "s1", [turn])

    resp = client.get("/api/sessions/s1/artifact/t0001/gradcam")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_chat_returns_503_when_api_key_missing(client, sessions_dir, monkeypatch):
    _write_session(sessions_dir, "s1", [_turn_dict()])
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    resp = client.post("/api/sessions/s1/chat", json={"message": "why was t0001 flagged?"})
    assert resp.status_code == 503


def test_chat_returns_502_with_gateway_detail_not_bare_500(client, sessions_dir):
    """Reproduces the real 2026-09-06 browser failure: the LLM Gateway rejected the
    request ("model qwen3.5-4b-32k-fast does not support tools") and the un-caught
    HTTPStatusError surfaced as a bare FastAPI 500 with no usable diagnostic. Same
    principle as ADR-027 — a failure must say what failed."""
    _write_session(sessions_dir, "s1", [_turn_dict()])

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"metadata": {"errors": ["model xyz does not support tools"]}, "code": 400},
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handle_request))

    def override_deps():
        return mock_client, AssemblyAISettings(api_key="test-key")

    backend_module.app.dependency_overrides[backend_module.get_analyst_agent_dependencies] = override_deps
    try:
        resp = client.post("/api/sessions/s1/chat", json={"message": "why?"})
    finally:
        backend_module.app.dependency_overrides.clear()

    assert resp.status_code == 502
    assert "does not support tools" in resp.json()["detail"]


def test_chat_returns_actionable_429_not_raw_gateway_json(client, sessions_dir):
    """Hit for real on the trial account mid-conversation. A raw
    {"message":"too many requests for this action"} dump in a chat bubble tells a
    reviewer nothing about what to do; the response must say to wait and retry."""
    _write_session(sessions_dir, "s1", [_turn_dict()])

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "too many requests for this action", "code": 429})

    mock_client = httpx.Client(transport=httpx.MockTransport(handle_request))

    def override_deps():
        return mock_client, AssemblyAISettings(api_key="test-key")

    backend_module.app.dependency_overrides[backend_module.get_analyst_agent_dependencies] = override_deps
    try:
        resp = client.post("/api/sessions/s1/chat", json={"message": "why?"})
    finally:
        backend_module.app.dependency_overrides.clear()

    assert resp.status_code == 429
    detail = resp.json()["detail"]
    assert "rate-limiting" in detail
    assert "Wait a few seconds" in detail


def test_chat_grounds_answer_in_real_tool_call(client, sessions_dir):
    _write_session(sessions_dir, "s1", [_turn_dict(turn_id="t0007", score=0.91)])

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not any(m.get("role") == "tool" for m in body["messages"]):
            # first call: model asks for the tool
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_turn_evidence",
                                            "arguments": json.dumps({"turn_id": "t0007"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        # second call: model answers using the tool result already in the transcript
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Turn t0007 scored 0.91."}}]},
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handle_request))

    def override_deps():
        return mock_client, AssemblyAISettings(api_key="test-key")

    backend_module.app.dependency_overrides[backend_module.get_analyst_agent_dependencies] = override_deps
    try:
        resp = client.post("/api/sessions/s1/chat", json={"message": "why was t0007 flagged?"})
    finally:
        backend_module.app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert "0.91" in resp.json()["answer"]


def test_markdown_renderer_is_served_and_loaded_before_app_js(client):
    """The chat panel renders the agent's Markdown via web/static/markdown.js. It must
    be served, and loaded BEFORE app.js — app.js calls renderMarkdown at message time,
    so a missing or late-loaded module breaks every agent reply with a ReferenceError.
    (The renderer's own behaviour is covered by its Node test; this guards the wiring.)"""
    resp = client.get("/markdown.js")
    assert resp.status_code == 200
    assert "renderMarkdown" in resp.text
    assert "escapeHtml" in resp.text  # HTML is escaped before Markdown is applied

    page = client.get("/index.html")
    assert page.status_code == 200
    assert page.text.index("markdown.js") < page.text.index("app.js")


def test_static_assets_are_sent_with_no_store(client):
    """A cached app.js silently masked a frontend fix — the file on disk was correct and
    served correctly, but the browser replayed the old one. For a local dev dashboard a
    stale asset hiding a fix costs more than caching saves."""
    for path in ("/app.js", "/markdown.js", "/style.css", "/"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "no-store" in resp.headers.get("cache-control", ""), path


def test_get_session_normalizes_legacy_turns_through_domain_layer(client, sessions_dir):
    """The endpoint used to return the raw file, so the dashboard and the Analyst Agent
    saw different data for the same session — legacy turns reached the UI with no
    `f0_detected`, `duration_ms` or `score_reliable` at all, while the agent got them
    via rehydration. One normalization path or they drift."""
    legacy = _turn_dict()
    legacy["prosody_raw"].pop("f0_detected", None)  # pre-ADR-026 shape
    _write_session(sessions_dir, "s1", [legacy], is_synthetic_demo=True)

    body = client.get("/api/sessions/s1").json()
    turn = body["turns"][0]

    assert body["is_synthetic_demo"] is True          # top-level metadata preserved
    assert "f0_detected" in turn["prosody_raw"]        # inferred, not missing
    assert turn["prosody_raw"]["f0_detected"] is True  # real pitch values -> True
    assert turn["stress"]["duration_ms"] == 2000       # computed, not stored
    assert turn["stress"]["score_reliable"] is True    # 2000ms is above the floor
