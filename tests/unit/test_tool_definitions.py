import pytest

from voicestress.domain.entities import InterviewSession, TurnEvidence
from voicestress.domain.value_objects import ArousalPrediction, ProsodyFeatures, SpeakerId
from voicestress.infrastructure.agents.tool_definitions import (
    ToolExecutionError,
    build_tool_handlers,
)


def _turn(turn_id, prob, baseline_dev=None) -> TurnEvidence:
    return TurnEvidence(
        turn_id=turn_id,
        session_id="s1",
        t_start_ms=0,
        t_end_ms=1000,
        is_baseline_turn=False,
        transcript=f"answer for {turn_id}",
        arousal=ArousalPrediction(probability=prob, model_version="v-test"),
        prosody_raw=ProsodyFeatures(180, 20, 1, 5, 15, 4),
        baseline_deviation=baseline_dev,
    )


@pytest.fixture
def session_with_turns() -> InterviewSession:
    session = InterviewSession(session_id="s1", speaker=SpeakerId("interview", "c1"))
    session.add_turn(_turn("t1", 0.3))
    session.add_turn(_turn("t2", 0.85, baseline_dev={"f0_mean_z": 2.1}))
    return session


def test_list_flagged_turns_respects_threshold(session_with_turns):
    handlers = build_tool_handlers(session_with_turns)
    result = handlers["list_flagged_turns"]({"min_score": 0.6})
    assert [t["turn_id"] for t in result["turns"]] == ["t2"]


def test_get_turn_evidence_returns_full_record(session_with_turns):
    handlers = build_tool_handlers(session_with_turns)
    result = handlers["get_turn_evidence"]({"turn_id": "t2"})
    assert result["turn_id"] == "t2"
    assert result["stress"]["score"] == pytest.approx(0.85)
    assert "disclaimer" in result


def test_get_turn_evidence_unknown_turn_raises(session_with_turns):
    handlers = build_tool_handlers(session_with_turns)
    with pytest.raises(ToolExecutionError):
        handlers["get_turn_evidence"]({"turn_id": "nope"})


def test_compare_to_baseline_reports_unreliable_explicitly(session_with_turns):
    handlers = build_tool_handlers(session_with_turns)
    result = handlers["compare_to_baseline"]({"turn_id": "t1"})  # no baseline_dev set
    assert result["baseline_reliable"] is False
    assert "message" in result


def test_compare_to_baseline_returns_deviation_when_available(session_with_turns):
    handlers = build_tool_handlers(session_with_turns)
    result = handlers["compare_to_baseline"]({"turn_id": "t2"})
    assert result["baseline_reliable"] is True
    assert result["deviation"]["f0_mean_z"] == pytest.approx(2.1)


def test_get_transcript_returns_verbatim_text(session_with_turns):
    handlers = build_tool_handlers(session_with_turns)
    result = handlers["get_transcript"]({"turn_id": "t1"})
    assert result["transcript"] == "answer for t1"
