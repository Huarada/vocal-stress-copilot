"""ADR-034: catch replies that announce an intention instead of answering."""
import httpx
import pytest

from voicestress.application.answer_validation import is_stalled_announcement
from voicestress.infrastructure.agents.llm_gateway_client import AnalystAgent


def test_detects_the_real_stalled_answer_verbatim():
    """The exact reply a reviewer got on 2026-09-06 — a narrated intention, no answer."""
    text = (
        "To explain the high arousal signal in turn **t0008**, I need to retrieve the "
        "specific acoustic evidence associated with that segment. Let me call "
        "`get_turn_evidence` for turn **t0008**."
    )
    assert is_stalled_announcement(text)


@pytest.mark.parametrize(
    "text",
    [
        "Let me call get_turn_evidence for that turn.",
        "I'll use compare_to_baseline to check this.",
        "First, I need to run list_flagged_turns.",
        "I'm going to make a tool call to fetch that record.",
    ],
)
def test_detects_stall_variants(text):
    assert is_stalled_announcement(text)


def test_real_answer_is_not_a_stall():
    text = (
        "Turn 8 scored 0.94. The model's attention concentrated around 376 Hz early in "
        "the clip, and pitch variability sat about 2 SD above this speaker's baseline."
    )
    assert not is_stalled_announcement(text)


def test_long_answer_mentioning_a_tool_in_passing_is_not_a_stall():
    """A complete answer that happens to name its source must not be flagged — only
    short replies that are *purely* an announcement."""
    text = (
        "Based on get_turn_evidence, turn 8 scored 0.94 with a duration of 2100ms, so "
        "the score is within the reliability floor. " + ("Detail. " * 80)
    )
    assert len(text) > 600
    assert not is_stalled_announcement(text)


def test_short_answer_without_retrieval_reference_is_not_a_stall():
    assert not is_stalled_announcement("Let me explain: arousal was elevated.")


def test_empty_answer_is_not_flagged_as_stall():
    assert not is_stalled_announcement("")


# --- agent integration --------------------------------------------------------------


def _agent(answers: list[str]) -> AnalystAgent:
    remaining = list(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": remaining.pop(0)}}]},
        )

    return AnalystAgent(
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        api_key="k",
        system_prompt="be careful",
        tool_schemas=[],
        tool_handlers={},
        grounding_mode="context",
        grounding_context="evidence",
    )


def test_stalled_answer_is_nudged_once_and_the_real_answer_returned():
    agent = _agent(
        [
            "Let me call `get_turn_evidence` for turn t0008.",
            "Turn 8 scored 0.94, with attention around 376 Hz early in the clip.",
        ]
    )
    answer = agent.ask("explain turn 8")
    assert answer.startswith("Turn 8 scored 0.94")
    assert agent.stalled_answers == 1


def test_good_first_answer_is_not_nudged():
    agent = _agent(["Turn 8 scored 0.94, attention around 376 Hz early in the clip."])
    agent.ask("explain turn 8")
    assert agent.stalled_answers == 0
