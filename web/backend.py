"""Dashboard backend (ARCHITECTURE.md §8's `web/`). Serves persisted interview sessions
(from `scripts/run_interview.py` or `scripts/build_demo_session.py`) to the frontend, and
proxies reviewer questions to the Analyst Agent.

Deliberately does NOT require ASSEMBLYAI_API_KEY to start or to serve session/turn data —
only the `/chat` endpoint needs it, and it fails that one request clearly (503) rather
than crashing the whole app, so the dashboard stays useful for reviewing past sessions
even with no key configured. That's what makes this file fully testable today (see
tests/unit/test_web_backend.py) despite Frente 1 (the live agent) not being.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# `web/` has no __init__.py — pytest can import this file's siblings (live_capture.py)
# flat because pyproject.toml puts `web/` on pythonpath for tests. Running the real
# server as `python -m uvicorn web.backend:app` imports THIS file as `web.backend` (a
# namespace-package submodule) instead, which does NOT add `web/` itself to sys.path —
# only the repo root is on it. `from live_capture import ...` then raised
# ModuleNotFoundError the first time this was actually run outside pytest (caught
# 2026-09-05, ADR-041). Making `web/`'s own directory sys.path-explicit here makes the
# import work identically under both invocation styles, rather than depending on
# whichever one happens to already have it on sys.path by accident.
_WEB_DIR = str(Path(__file__).resolve().parent)
if _WEB_DIR not in sys.path:
    sys.path.insert(0, _WEB_DIR)

import httpx
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voicestress.config import AssemblyAISettings, MissingConfigError
from voicestress.domain.entities import InterviewSession
from voicestress.domain.value_objects import SpeakerId
from voicestress.infrastructure.agents.llm_gateway_client import AnalystAgent, AnalystAgentError
from voicestress.infrastructure.agents.tool_definitions import (
    ALL_TOOL_SCHEMAS,
    build_tool_handlers,
    render_session_evidence,
)

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "artifacts" / "sessions"
PROMPT_PATH = ROOT / "agents" / "prompts" / "analyst.md"
MODEL_PATH = ROOT / "artifacts" / "models" / "arousal_resnet_light.keras"

def _env_flag(name: str, default: bool = False) -> bool:
    """Case-insensitive boolean env var read, pulled out as its own function so the
    parsing itself is testable without reloading this module (which re-registers every
    route on a fresh FastAPI instance — real side effects, not worth risking for a
    one-line parse)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


# ADR-049: live capture and the Analyst Agent chat share one ASSEMBLYAI_API_KEY, but
# they are NOT the same cost exposure — a continuous, per-minute-billed voice session
# vs. a bounded, at-most-two-calls text request with existing 429 handling and no card
# on file. Configuring the key alone would enable both at once; this flag lets a
# deployment enable chat (so a reviewer can actually use the Analyst Agent on the
# hosted URL) while keeping live capture off regardless of whether a key is present.
# Defaults to disabled — a deployment must opt in explicitly, not opt out.
LIVE_CAPTURE_ENABLED = _env_flag("VOICESTRESS_ENABLE_LIVE_CAPTURE")

app = FastAPI(title="Voice Stress Co-Pilot — Dashboard")


@app.middleware("http")
async def no_store_static(request, call_next):
    """Stop the browser caching the dashboard's JS/CSS.

    Added 2026-09-06 after a frontend fix (Markdown rendering in the chat) appeared not
    to work: the file on disk was correct and served correctly, but the browser kept
    replaying a cached app.js. For a locally-run dev dashboard, a stale asset silently
    masking a fix costs far more than the caching saves.

    Note this does NOT help with Python-side changes — uvicorn holds imported modules in
    memory, so backend edits need a restart (or `--reload`).
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


class ChatRequest(BaseModel):
    message: str


def _session_path(session_id: str) -> Path:
    # session_id comes straight off the URL path; keep it to the exact filename shape
    # this app itself writes (no path separators) rather than trusting it as a path.
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="Invalid session id")
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return path


def load_session_raw(session_id: str) -> dict:
    return json.loads(_session_path(session_id).read_text(encoding="utf-8"))


def load_session_entity(session_id: str) -> InterviewSession:
    raw = load_session_raw(session_id)
    return InterviewSession.from_session_dict(raw, speaker=SpeakerId("dashboard", session_id))


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    if not SESSIONS_DIR.exists():
        return []
    results = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results.append(
            {
                "session_id": data.get("session_id", path.stem),
                "n_turns": len(data.get("turns", [])),
                "is_synthetic_demo": data.get("is_synthetic_demo", False),
                "note": data.get("note"),
            }
        )
    return results


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """Returns the session with every turn NORMALIZED through the domain layer, not the
    raw file.

    This used to return the file verbatim, which quietly gave the dashboard and the
    Analyst Agent two different views of the same session: the agent reads rehydrated
    entities (so it gets ADR-026 legacy pitch inference, `duration_ms`,
    `score_reliable`, Grad-CAM summaries), while the UI got whatever the file happened
    to contain — legacy sessions arrived with no `f0_detected` at all. One normalization
    path, so what a reviewer sees and what the agent reasons over cannot drift apart.
    """
    raw = load_session_raw(session_id)
    session = load_session_entity(session_id)
    return {
        **{k: v for k, v in raw.items() if k != "turns"},
        "turns": [turn.to_evidence_dict() for turn in session.turns],
    }


@app.get("/api/sessions/{session_id}/artifact/{turn_id}/{kind}")
def get_turn_artifact(session_id: str, turn_id: str, kind: str):
    if kind not in {"spectrogram", "gradcam"}:
        raise HTTPException(status_code=400, detail="kind must be 'spectrogram' or 'gradcam'")

    session = load_session_entity(session_id)
    turn = session.get_turn(turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail=f"Turn {turn_id!r} not found")

    path_str = turn.gradcam_path if kind == "gradcam" else turn.spectrogram_path
    if not path_str:
        raise HTTPException(status_code=404, detail=f"No {kind} artifact recorded for this turn")

    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact file missing on disk: {path}")
    return FileResponse(path, media_type="image/png")


def get_analyst_agent_dependencies() -> tuple[httpx.Client, AssemblyAISettings]:
    """Overridden in tests to inject a mock httpx client — see
    tests/unit/test_web_backend.py — so the chat endpoint's tool-calling wiring is
    testable without a live LLM Gateway connection."""
    settings = AssemblyAISettings.from_env()
    return httpx.Client(), settings


@app.post("/api/sessions/{session_id}/chat")
def chat(
    session_id: str,
    body: ChatRequest,
    deps: tuple[httpx.Client, AssemblyAISettings] = Depends(get_analyst_agent_dependencies),
) -> dict:
    session = load_session_entity(session_id)
    http_client, settings = deps

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    grounding_mode = settings.llm_grounding_mode
    agent = AnalystAgent(
        http_client=http_client,
        api_key=settings.api_key,
        system_prompt=system_prompt,
        tool_schemas=ALL_TOOL_SCHEMAS,
        tool_handlers=build_tool_handlers(session),
        base_url=settings.llm_gateway_base_url,
        grounding_mode=grounding_mode,
        # Only rendered when actually needed — see ADR-030 for why "context" is the
        # default on this account (its one reachable model rejects `tools`).
        grounding_context=(
            render_session_evidence(session) if grounding_mode == "context" else None
        ),
    )

    try:
        answer = agent.ask(body.message)
    except httpx.HTTPStatusError as e:
        # ADDED 2026-09-06 after a real HTTP 500 in the browser with no usable
        # diagnostic: an un-caught HTTPStatusError from the gateway surfaced as a bare
        # FastAPI 500, hiding the actual cause ("model qwen3.5-4b-32k-fast does not
        # support tools"). Same principle as ADR-027 — a failure must say what failed.
        status = e.response.status_code if e.response is not None else 0

        if status == 429:
            # Hit for real on the trial account. Raw gateway JSON in a chat bubble is
            # not an answer a reviewer can act on; say what happened and what to do.
            # Note the vocabulary guard (ADR-031) doubles request count on a violation,
            # which makes this more likely — a documented cost of that guard.
            raise HTTPException(
                status_code=429,
                detail=(
                    "The LLM Gateway is rate-limiting this account (HTTP 429). Wait a "
                    "few seconds and ask again. Free-tier accounts have a low request "
                    "ceiling, and a refused answer costs an extra request because the "
                    "agent retries once when it breaks its own vocabulary rules."
                ),
            ) from e

        detail = e.response.text[:500] if e.response is not None else str(e)
        raise HTTPException(
            status_code=502,
            detail=f"LLM Gateway rejected the request (HTTP {status}): {detail}",
        ) from e
    except AnalystAgentError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {"answer": answer}


def get_voice_agent_connect():
    """Overridden in tests (see tests/unit/test_live_capture.py) to inject a fake
    AssemblyAI transport — the same dependency-override technique
    get_analyst_agent_dependencies uses for /chat, applied to a WebSocket route so this
    endpoint's full lifecycle is testable with no live ASSEMBLYAI_API_KEY. `None` here
    means "use the real connection"; LiveInterviewRunner.run() supplies that default."""
    return None


@app.websocket("/ws/interview")
async def interview_capture(
    ws: WebSocket, connect=Depends(get_voice_agent_connect)
) -> None:
    """Browser-based capture (ADR-041): a candidate's mic audio in, the Interview
    Agent's transcript/reply audio out, live turn evidence out for whoever is watching
    the page. See web/live_capture.py for the framing and orchestration — this route
    only resolves settings and hands off, so live_capture.py stays independent of how
    a caller obtains its AssemblyAISettings (tests construct one directly).

    MissingConfigError is caught HERE, not via the `@app.exception_handler` below —
    that handler wraps HTTP responses and does not apply to a WebSocket's connection
    handshake, which has no equivalent response channel until after `ws.accept()`.

    `run_capture_session` is imported LAZILY, right before use, not at module top —
    it pulls in `LiveInterviewRunner`, which pulls in TensorFlow (for the arousal
    classifier). A pure dashboard deployment with no ASSEMBLYAI_API_KEY configured
    (e.g. a low-memory public host that only shows the demo session, ADR-046) never
    reaches this line, so it never pays TensorFlow's import time or its 500MB+ memory
    footprint — the `except MissingConfigError` branch above returns first.

    LIVE_CAPTURE_ENABLED (ADR-049) is checked BEFORE the API key at all — a deployment
    can have a real key configured (to power /chat) while this stays unset, and live
    capture is refused regardless of the key's presence. Kept as its own explicit
    branch, not folded into the missing-key path, so the two reasons for refusal read
    differently to whoever's watching the log: "not configured" vs. "disabled here."
    """
    if not LIVE_CAPTURE_ENABLED:
        await ws.accept()
        await ws.send_json(
            {
                "type": "error",
                "message": (
                    "Live capture is disabled on this deployment. Run the project "
                    "locally to try it (see README.md)."
                ),
            }
        )
        await ws.close(code=1008)
        return
    try:
        settings = AssemblyAISettings.from_env()
    except MissingConfigError as e:
        await ws.accept()
        await ws.send_json(
            {"type": "error", "message": f"Voice Agent not configured: {e}"}
        )
        await ws.close(code=1008)
        return
    from live_capture import run_capture_session

    await run_capture_session(ws, settings=settings, model_path=MODEL_PATH, connect=connect)


@app.exception_handler(MissingConfigError)
def handle_missing_config(request, exc: MissingConfigError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=503,
        content={"detail": f"Analyst Agent not configured: {exc}"},
    )


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
