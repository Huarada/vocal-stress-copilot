"""AssemblyAI LLM Gateway client (ARCHITECTURE.md §5) — the Analyst Agent adapter.

OpenAI-compatible chat-completions endpoint with tool calling (verified against
https://www.assemblyai.com/docs/llm-gateway/overview on 2026-09-04, SKILLS.md §3). This
module runs the standard tool-calling loop: send messages + tool schemas, and whenever
the model responds with tool calls instead of text, execute them locally (via the
handlers from `tool_definitions.build_tool_handlers`) and feed the results back, until
the model produces a final text answer.

httpx.Client is injected (as `http_client`) rather than constructed inside — this is what
lets tests drive the whole loop with `httpx.MockTransport`, no network, no API key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from voicestress.application.answer_validation import (
    CONTINUATION_INSTRUCTION,
    is_stalled_announcement,
)
from voicestress.application.vocabulary_guard import (
    CORRECTION_INSTRUCTION,
    SAFE_FALLBACK_ANSWER,
    contains_violation,
)

DEFAULT_MODEL = "qwen3.5-4b-32k-fast"  # confirmed WORKING against a real free-trial
# account on 2026-09-05 (scripts/check_live_connection.py). claude-sonnet-5,
# claude-haiku-4-5-20251001 and gpt-4.1 are all listed on the LLM Gateway's public
# model roster but returned HTTP 400 "Your account does not have access to this LLM
# Gateway model" on this account's $150 free-trial tier — the roster page documents
# what CAN exist behind the gateway, not what a given account is entitled to. If your
# account has a paid/upgraded plan, override via AnalystAgent(model="claude-sonnet-5").
MAX_TOOL_CALL_ROUNDS = 6  # guards against a runaway tool-call loop


class AnalystAgentError(RuntimeError):
    pass


@dataclass
class AnalystAgent:
    """One conversation with the Analyst Agent. `tool_handlers` comes from
    `tool_definitions.build_tool_handlers(session)` — this class doesn't know about
    `InterviewSession` at all, only about executing named tools against a handler map,
    which keeps it testable independent of the domain layer's evidence store.
    """

    http_client: httpx.Client
    api_key: str
    system_prompt: str
    tool_schemas: list[dict[str, Any]]
    tool_handlers: dict[str, Callable[[dict], Any]]
    base_url: str = "https://llm-gateway.assemblyai.com"
    model: str = DEFAULT_MODEL

    # GROUNDING MODE (ADR-030). "tools" = the model calls tools on demand (needs a
    # model with function-calling support). "context" = the caller pre-renders the
    # session's evidence into `grounding_context` and no tools are sent at all —
    # required on accounts whose only accessible model doesn't support tools, which is
    # exactly the case on this project's trial account: `qwen3.5-4b-32k-fast` returns
    # HTTP 400 "model does not support tools", and every other model on the public
    # roster returns "your account does not have access". Both modes preserve ADR-004's
    # actual guarantee — the model can only speak about data we put in front of it.
    grounding_mode: str = "tools"
    grounding_context: str | None = None

    messages: list[dict[str, Any]] = field(default_factory=list)
    # Counts how many times the vocabulary guard had to intervene this conversation —
    # surfaced so a reviewer/operator can see the model is fighting its constraints
    # rather than having that fact silently absorbed (ADR-027's principle again).
    vocabulary_violations: int = 0
    # Counts replies that announced an intention instead of answering (ADR-034).
    stalled_answers: int = 0

    def __post_init__(self) -> None:
        if self.grounding_mode not in ("tools", "context"):
            raise ValueError(f"grounding_mode must be 'tools' or 'context', got {self.grounding_mode!r}")
        if self.grounding_mode == "context" and not self.grounding_context:
            raise ValueError("grounding_mode='context' requires grounding_context to be set")

        if not self.messages:
            self.messages = [{"role": "system", "content": self._build_system_message()}]

    def _build_system_message(self) -> str:
        """`agents/prompts/analyst.md` is deliberately mode-neutral about *how* evidence
        arrives; this appends the mechanism.

        Added 2026-09-06 after a real failure: the prompt still said "call
        `get_turn_evidence`", so in context mode the model dutifully replied *"Let me
        call `get_turn_evidence` for turn t0008"* — announcing a tool call it had no
        tools to make — and stopped, producing no answer at all. Telling a model to use
        a mechanism that isn't there is a prompt bug, not a model failure.
        """
        if self.grounding_mode == "tools":
            access = (
                "=== ACCESS MODE: TOOLS ===\n"
                "Retrieve evidence by calling the tools provided to you "
                "(`list_flagged_turns`, `get_turn_evidence`, `compare_to_baseline`, "
                "`get_transcript`). State nothing that did not come back from a call you "
                "actually made."
            )
            return f"{self.system_prompt}\n\n{access}"

        access = (
            "=== ACCESS MODE: INLINE EVIDENCE ===\n"
            "You have NO tools in this deployment. The complete session evidence is "
            "already included below — it is the only data you may discuss. Do NOT "
            "announce, request, or describe calling any tool or function (there are none "
            "to call, and saying you will call one produces no answer for the reviewer). "
            "Read the evidence below and answer directly and completely in a single "
            "reply.\n\n"
            "=== SESSION EVIDENCE ===\n"
            f"{self.grounding_context}"
        )
        return f"{self.system_prompt}\n\n{access}"

    def ask(self, user_message: str) -> str:
        answer = self._ask_once(user_message)

        # STALL GUARD (ADR-034): a real answer was just "Let me call `get_turn_evidence`
        # for turn t0008." and nothing else — an announced intention instead of an
        # explanation. Nudge once; a stall is a non-answer, not a harmful one, so
        # there's no reason to refuse outright if the nudge works.
        if is_stalled_announcement(answer):
            self.stalled_answers += 1
            answer = self._ask_once(CONTINUATION_INSTRUCTION)

        # VOCABULARY GUARD (ADR-031/ADR-033): the prompt-level rule is not enough with a
        # small model — a real live answer speculated about the speaker "suppressing
        # their lie". Detect, give the model exactly one corrective attempt, then refuse
        # rather than render a violating answer.
        if not contains_violation(answer):
            return answer

        self.vocabulary_violations += 1
        corrected = self._ask_once(CORRECTION_INSTRUCTION)
        if not contains_violation(corrected):
            return corrected

        self.vocabulary_violations += 1
        return SAFE_FALLBACK_ANSWER

    def _ask_once(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(MAX_TOOL_CALL_ROUNDS):
            response = self._chat_completion()
            choice = response["choices"][0]["message"]
            self.messages.append(choice)

            tool_calls = choice.get("tool_calls")
            if not tool_calls:
                return choice.get("content", "")

            for call in tool_calls:
                self._execute_and_record_tool_call(call)

        raise AnalystAgentError(
            f"Exceeded {MAX_TOOL_CALL_ROUNDS} tool-call rounds without a final answer — "
            "likely a tool-calling loop; inspect self.messages."
        )

    def _chat_completion(self) -> dict:
        payload: dict[str, Any] = {"model": self.model, "messages": self.messages}
        if self.grounding_mode == "tools":
            payload["tools"] = [
                {"type": "function", "function": schema} for schema in self.tool_schemas
            ]
        response = self.http_client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        return response.json()

    def _execute_and_record_tool_call(self, call: dict[str, Any]) -> None:
        function = call["function"]
        name = function["name"]
        arguments = json.loads(function["arguments"]) if function.get("arguments") else {}

        handler = self.tool_handlers.get(name)
        if handler is None:
            result: Any = {"error": f"Unknown tool: {name}"}
        else:
            try:
                result = handler(arguments)
            except Exception as e:  # noqa: BLE001 - surfaced to the model, not swallowed
                result = {"error": str(e)}

        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result),
            }
        )
