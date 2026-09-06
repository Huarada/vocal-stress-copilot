"""Drives the Analyst Agent's tool-calling loop against httpx.MockTransport — no
network, no API key. Mirrors how a real LLM Gateway response is shaped (OpenAI-
compatible chat completions with tool_calls), scripted here instead of live."""
import json

import httpx
import pytest

from voicestress.infrastructure.agents.llm_gateway_client import AnalystAgent, AnalystAgentError


def _text_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _tool_call_response(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


def _make_agent(handler, responses: list[dict]) -> AnalystAgent:
    responses_iter = iter(responses)

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses_iter))

    client = httpx.Client(transport=httpx.MockTransport(handle_request))
    return AnalystAgent(
        http_client=client,
        api_key="test-key",
        system_prompt="You explain vocal stress signals. Never say 'lying'.",
        tool_schemas=[{"name": "get_turn_evidence", "parameters": {}}],
        tool_handlers={"get_turn_evidence": handler},
    )


def test_direct_text_answer_no_tool_calls():
    agent = _make_agent(handler=lambda args: {}, responses=[_text_response("Hello, reviewer.")])
    answer = agent.ask("hi")
    assert answer == "Hello, reviewer."


def test_tool_call_then_final_answer():
    calls_made = []

    def handler(args):
        calls_made.append(args)
        return {"turn_id": args["turn_id"], "stress": {"score": 0.71}}

    agent = _make_agent(
        handler=handler,
        responses=[
            _tool_call_response("call_1", "get_turn_evidence", {"turn_id": "t7"}),
            _text_response("Turn t7 showed elevated vocal arousal (0.71) vs. baseline."),
        ],
    )

    answer = agent.ask("Why was turn 7 flagged?")

    assert calls_made == [{"turn_id": "t7"}]
    assert "0.71" in answer
    assert "lying" not in answer.lower()

    # tool result must have been recorded in the message history, correctly tagged
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_messages[0]["content"])["stress"]["score"] == 0.71


def test_unknown_tool_name_reports_error_without_crashing():
    agent = _make_agent(
        handler=lambda args: {},
        responses=[
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {"name": "nonexistent_tool", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            },
            _text_response("I couldn't find that information."),
        ],
    )
    answer = agent.ask("test")
    assert answer == "I couldn't find that information."
    tool_result = json.loads(agent.messages[-2]["content"])
    assert "error" in tool_result


def test_handler_exception_is_surfaced_as_tool_error_not_raised():
    def failing_handler(args):
        raise ValueError("no such turn")

    agent = _make_agent(
        handler=failing_handler,
        responses=[
            _tool_call_response("call_1", "get_turn_evidence", {"turn_id": "ghost"}),
            _text_response("That turn does not exist."),
        ],
    )
    answer = agent.ask("explain turn ghost")
    assert answer == "That turn does not exist."


def test_runaway_tool_loop_raises_instead_of_looping_forever():
    responses = [
        _tool_call_response(f"call_{i}", "get_turn_evidence", {"turn_id": "t1"}) for i in range(10)
    ]
    agent = _make_agent(handler=lambda args: {"ok": True}, responses=responses)
    with pytest.raises(AnalystAgentError):
        agent.ask("loop forever")
