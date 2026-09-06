"""Live connectivity diagnostic — NOT a full interview run. Validates that
ASSEMBLYAI_API_KEY authenticates correctly against the real Voice Agent API and LLM
Gateway, without opening a microphone or recording anyone. This is the safe first check
before ever running `scripts/run_interview.py` for real: if auth or the protocol
handshake is broken, better to find out here than mid-interview.

Also the tool that resolves the `# ASSUMPTION:` comments in run_interview.py — every
event this script logs is real, observed protocol behavior, not documentation prose.

Usage:
    python scripts/check_live_connection.py
"""
from __future__ import annotations

import asyncio
import json

import httpx

from voicestress.config import AssemblyAISettings
from voicestress.infrastructure.agents.tool_definitions import ALL_TOOL_SCHEMAS
from voicestress.infrastructure.agents.voice_agent_client import VoiceAgentSession
from voicestress.infrastructure.agents.websocket_transport import WebsocketTransport

CONNECT_TIMEOUT_S = 15


async def check_voice_agent(settings: AssemblyAISettings) -> None:
    print(f"\n=== Voice Agent API: {settings.voice_agent_ws_url} ===")
    try:
        transport = await asyncio.wait_for(
            WebsocketTransport.connect(settings.voice_agent_ws_url, settings.api_key),
            timeout=CONNECT_TIMEOUT_S,
        )
    except Exception as e:
        print(f"[FAIL] could not open websocket / auth rejected: {type(e).__name__}: {e}")
        return

    events_seen: list[str] = []

    async def on_ready(event):
        events_seen.append("session.ready")
        print(f"[event] session.ready — session_id={event.get('session_id')}")

    async def on_updated(event):
        events_seen.append("session.updated")
        config = event.get("config") or {}
        print(f"[event] session.updated — session_id={config.get('id')}")

    async def on_error(event):
        events_seen.append("session.error")
        print(f"[event] session.error — {event}")

    async def dump_raw_event(label: str) -> dict:
        try:
            event = await asyncio.wait_for(session.handle_next_event(), timeout=CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError:
            print(f"[FAIL] no event received within {CONNECT_TIMEOUT_S}s ({label})")
            return {}
        print(f"[event, {label}] type={event.get('type')!r} raw={json.dumps(event)[:300]}")
        return event

    async with transport:
        session = VoiceAgentSession(
            transport=transport,
            on_session_ready=on_ready,
            on_session_updated=on_updated,
            on_error=on_error,
        )

        # CONFIRMED 2026-09-05: this account's connection does NOT emit an event before
        # session.update is sent — the wait below reliably times out. Kept (rather than
        # removed) so a future account/config that DOES greet first is still observed
        # and logged instead of silently skipped.
        await dump_raw_event("post-connect, before session.update")

        await session.configure(
            tools=[],
            system_prompt="You are a connectivity test. Say nothing unprompted.",
        )
        print("[sent] session.update")

        await dump_raw_event("after session.update")

        if "session.ready" in events_seen or "session.updated" in events_seen:
            print(f"[OK] authenticated, session configured (session_id={session.session_id})")
        elif "session.error" in events_seen:
            print("[FAIL] server returned session.error — see above")
        else:
            print("[?] no recognized session event seen in either slot — see raw event dumps above")

        await session.end_session()
        print("[sent] session.end")


DIAGNOSTIC_TOOL_SCHEMA = {
    "type": "function",
    "name": "note_test_signal",
    "description": "Test-only diagnostic tool. Call this immediately, before saying anything else.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


async def check_voice_agent_tool_calling(settings: AssemblyAISettings) -> None:
    """Tool-calling on the Voice Agent API is a SEPARATE entitlement path from LLM
    Gateway chat-completions tool-calling (check_tool_calling_support above) — the
    Voice Agent product routes to its own internal model, not one picked by name from
    the LLM Gateway's roster. A negative result there says nothing about this path,
    which has never been tested. Configures a session with one harmless diagnostic
    tool and a system prompt that instructs calling it immediately, then watches for a
    real `tool.call` event."""
    print(f"\n=== Voice Agent API tool-calling: {settings.voice_agent_ws_url} ===")
    try:
        transport = await asyncio.wait_for(
            WebsocketTransport.connect(settings.voice_agent_ws_url, settings.api_key),
            timeout=CONNECT_TIMEOUT_S,
        )
    except Exception as e:
        print(f"[FAIL] could not open websocket / auth rejected: {type(e).__name__}: {e}")
        return

    tool_called = asyncio.Event()
    call_seen: dict = {}

    async def on_tool_call(event):
        call_seen.update(event)
        tool_called.set()

    async def on_error(event):
        print(f"[event] session.error — {event}")

    async with transport:
        session = VoiceAgentSession(
            transport=transport, on_tool_call=on_tool_call, on_error=on_error
        )
        await session.configure(
            tools=[DIAGNOSTIC_TOOL_SCHEMA],
            system_prompt=(
                "You are a diagnostic test with exactly one job: call the "
                "note_test_signal tool with value='ok' as your very first action. Do "
                "not say anything before calling it."
            ),
        )
        print("[sent] session.update with one diagnostic tool")
        await session.request_reply()
        print("[sent] reply.create")

        try:
            while not tool_called.is_set():
                event = await asyncio.wait_for(session.handle_next_event(), timeout=CONNECT_TIMEOUT_S)
                if event.get("type") not in ("tool.call",):
                    print(f"[event] type={event.get('type')!r}")
        except asyncio.TimeoutError:
            print(f"[FAIL] no tool.call within {CONNECT_TIMEOUT_S}s — see events above")
        else:
            print(f"[OK] tool.call received: {call_seen}")

        await session.end_session()
        print("[sent] session.end")


CANDIDATE_MODELS = [
    "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101", "qwen3.5-4b-32k-fast",
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-4-5-20250929", "claude-sonnet-4-6", "claude-sonnet-5",
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3.1-flash-lite",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "gemma-4-31b", "gpt-oss-120b", "gpt-oss-20b", "gpt-4.1", "gpt-5",
    "gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.5", "gpt-5.6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "qwen3-32B", "qwen3-next-80b-a3b",
]  # ADR-024's original 4-model guess replaced 2026-09-06 with the full live roster
# from GET /v1/models (list_available_models below) — no more guessing model names.


def list_available_models(settings: AssemblyAISettings) -> None:
    """OpenAI-compatible gateways conventionally expose `GET /v1/models`. Not
    documented for this gateway specifically — this is a probe, not an assumption — but
    if it responds, it's a live, current roster instead of guessing new model name
    strings the way CANDIDATE_MODELS above was originally built (ADR-024)."""
    print(f"\n=== LLM Gateway model roster: {settings.llm_gateway_base_url}/v1/models ===")
    try:
        with httpx.Client(timeout=CONNECT_TIMEOUT_S) as client:
            response = client.get(
                f"{settings.llm_gateway_base_url}/v1/models",
                headers={"Authorization": f"Bearer {settings.api_key}"},
            )
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return

    if response.status_code != 200:
        print(f"[not available] HTTP {response.status_code} — {response.text[:200]}")
        return

    try:
        model_ids = [m.get("id") for m in response.json().get("data", [])]
    except Exception as e:
        print(f"[FAIL] unexpected response shape: {type(e).__name__}: {e} — raw: {response.text[:300]}")
        return

    print(f"[OK] {len(model_ids)} model(s) listed: {model_ids}")


def check_llm_gateway(settings: AssemblyAISettings) -> None:
    print(f"\n=== LLM Gateway: {settings.llm_gateway_base_url} ===")
    with httpx.Client(timeout=CONNECT_TIMEOUT_S) as client:
        for model in CANDIDATE_MODELS:
            try:
                response = client.post(
                    f"{settings.llm_gateway_base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                        "max_tokens": 5,
                    },
                )
            except Exception as e:
                print(f"[{model}] [FAIL] {type(e).__name__}: {e}")
                continue

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                print(f"[{model}] [OK] HTTP 200 — replied: {content!r}")
            else:
                print(f"[{model}] [FAIL] HTTP {response.status_code} — {response.text[:200]}")


def check_tool_calling_support(settings: AssemblyAISettings) -> None:
    """ADR-024 established that plain chat completion access does not imply tool-
    calling access — the account's only reachable model at the time
    (qwen3.5-4b-32k-fast) returned HTTP 400 'model does not support tools' specifically,
    while replying fine to a bare chat message. That distinction is why
    `AnalystAgent`'s production `grounding_mode` defaults to "context", not "tools" —
    the code path for tool-calling (llm_gateway_client.py's OpenAI-style `{"type":
    "function", "function": schema}` wrapping) has never actually fired against a live
    model as a result. This re-probes with a REAL production tool schema (not a
    synthetic one) so a positive result here is directly actionable: switch
    `ASSEMBLYAI_LLM_GROUNDING_MODE=tools` and it should just work.
    """
    print(f"\n=== LLM Gateway tool-calling support: {settings.llm_gateway_base_url} ===")
    tools_payload = [{"type": "function", "function": schema} for schema in ALL_TOOL_SCHEMAS]

    with httpx.Client(timeout=CONNECT_TIMEOUT_S) as client:
        for model in CANDIDATE_MODELS:
            try:
                response = client.post(
                    f"{settings.llm_gateway_base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.api_key}"},
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Call get_transcript for turn_id 't0001'.",
                            }
                        ],
                        "tools": tools_payload,
                        "max_tokens": 100,
                    },
                )
            except Exception as e:
                print(f"[{model}] [FAIL] {type(e).__name__}: {e}")
                continue

            if response.status_code == 200:
                message = response.json()["choices"][0]["message"]
                called = bool(message.get("tool_calls"))
                print(
                    f"[{model}] [OK] HTTP 200, tools accepted — "
                    f"{'model actually called a tool' if called else 'model replied without calling one (schema accepted either way)'}"
                )
            else:
                # The specific text to look for: "does not support tools" means the
                # model itself lacks the capability, distinct from other 400 causes
                # (e.g. no account access at all, already reported by check_llm_gateway).
                print(f"[{model}] [FAIL] HTTP {response.status_code} — {response.text[:200]}")


async def main() -> None:
    settings = AssemblyAISettings.from_env()
    await check_voice_agent(settings)
    await check_voice_agent_tool_calling(settings)
    list_available_models(settings)
    check_llm_gateway(settings)
    check_tool_calling_support(settings)


if __name__ == "__main__":
    asyncio.run(main())
