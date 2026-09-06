"""Environment-based settings. Nothing in this module talks to a network — it only
reads env vars and validates presence, so it's importable and testable without secrets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Loads .env from the repo root into os.environ, once, at import time — this is
    # what makes `ASSEMBLYAI_API_KEY=...` in .env actually reach `from_env()` below
    # without every script having to `export` it manually first. Never overrides an
    # already-set real environment variable (override=False).
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=False)
except ImportError:  # python-dotenv not installed — fall back to real env vars only
    pass


class MissingConfigError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AssemblyAISettings:
    api_key: str
    voice_agent_ws_url: str = "wss://agents.assemblyai.com/v1/ws"
    streaming_stt_ws_url: str = "wss://streaming.assemblyai.com/v3/ws"
    llm_gateway_base_url: str = "https://llm-gateway.assemblyai.com"
    # See ADR-030. Defaults to "context" because this project's trial account can only
    # reach `qwen3.5-4b-32k-fast`, which the gateway rejects with HTTP 400 "model does
    # not support tools". Set ASSEMBLYAI_LLM_GROUNDING_MODE=tools on an account with a
    # tool-capable model to get on-demand retrieval instead of whole-session injection.
    llm_grounding_mode: str = "context"

    @staticmethod
    def from_env() -> "AssemblyAISettings":
        api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
        if not api_key:
            raise MissingConfigError(
                "ASSEMBLYAI_API_KEY is not set. Copy .env.example to .env and fill it in "
                "(see the account link in the hackathon brief), or export it directly."
            )
        return AssemblyAISettings(
            api_key=api_key,
            voice_agent_ws_url=os.environ.get(
                "ASSEMBLYAI_VOICE_AGENT_WS_URL", "wss://agents.assemblyai.com/v1/ws"
            ),
            streaming_stt_ws_url=os.environ.get(
                "ASSEMBLYAI_STREAMING_WS_URL", "wss://streaming.assemblyai.com/v3/ws"
            ),
            llm_gateway_base_url=os.environ.get(
                "ASSEMBLYAI_LLM_GATEWAY_URL", "https://llm-gateway.assemblyai.com"
            ),
            llm_grounding_mode=os.environ.get("ASSEMBLYAI_LLM_GROUNDING_MODE", "context"),
        )
