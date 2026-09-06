import pytest

from voicestress.domain.entities import InterviewSession, TurnEvidence
from voicestress.domain.value_objects import ArousalPrediction, ProsodyFeatures, SpeakerId


def _turn(session_id="s1", turn_id="t1", prob=0.7, baseline=False) -> TurnEvidence:
    return TurnEvidence(
        turn_id=turn_id,
        session_id=session_id,
        t_start_ms=0,
        t_end_ms=1000,
        is_baseline_turn=baseline,
        transcript="hello",
        arousal=ArousalPrediction(probability=prob, model_version="v-test"),
        prosody_raw=ProsodyFeatures(
            f0_mean_hz=180, f0_std_hz=20, jitter_local_pct=1, shimmer_local_pct=5,
            hnr_db=15, speech_rate_syll_s=4,
        ),
        baseline_deviation=None,
    )


def test_add_turn_rejects_mismatched_session():
    session = InterviewSession(session_id="s1", speaker=SpeakerId("interview", "c1"))
    with pytest.raises(ValueError):
        session.add_turn(_turn(session_id="OTHER"))


def test_baseline_vs_scored_turn_partition():
    session = InterviewSession(session_id="s1", speaker=SpeakerId("interview", "c1"))
    session.add_turn(_turn(turn_id="t0", baseline=True))
    session.add_turn(_turn(turn_id="t1", baseline=False))
    assert [t.turn_id for t in session.baseline_turns] == ["t0"]
    assert [t.turn_id for t in session.scored_turns] == ["t1"]


def test_flagged_turns_respects_threshold():
    session = InterviewSession(session_id="s1", speaker=SpeakerId("interview", "c1"))
    session.add_turn(_turn(turn_id="low", prob=0.2))
    session.add_turn(_turn(turn_id="high", prob=0.8))
    flagged = session.flagged_turns(min_score=0.6)
    assert [t.turn_id for t in flagged] == ["high"]


def test_duration_and_reliability_flag_short_clip():
    """Reproduces the real 2026-09-06 finding: turns under the spectrogram extractor's
    1s padding threshold get zero-padded and score erratically (0.67-0.97 regardless
    of content), while longer turns stayed in a lower, more stable band. Short turns
    must be flagged unreliable rather than presented at face value."""
    turn = _turn(prob=0.95)
    turn.t_start_ms, turn.t_end_ms = 0, 400  # 400ms — well under the 1000ms threshold
    assert turn.duration_ms == 400
    assert turn.arousal_score_reliable is False


def test_duration_and_reliability_flag_long_clip():
    turn = _turn(prob=0.3)
    turn.t_start_ms, turn.t_end_ms = 500, 2400  # 1900ms
    assert turn.duration_ms == 1900
    assert turn.arousal_score_reliable is True


def test_evidence_dict_carries_reliability_flag():
    turn = _turn(prob=0.9)
    turn.t_start_ms, turn.t_end_ms = 0, 300
    d = turn.to_evidence_dict()
    assert d["stress"]["duration_ms"] == 300
    assert d["stress"]["score_reliable"] is False


def test_get_turn_returns_none_when_missing():
    session = InterviewSession(session_id="s1", speaker=SpeakerId("interview", "c1"))
    assert session.get_turn("nope") is None


def test_to_evidence_dict_never_uses_deception_vocabulary():
    """Hard rule check (SKILLS.md #1): the evidence dict — the exact object the
    Analyst Agent's tools return — must never contain the deception vocabulary in any
    of its own field values (the disclaimer text is explicitly allowed to explain the
    exclusion, so we check the OTHER fields)."""
    turn = _turn(prob=0.9)
    d = turn.to_evidence_dict()
    banned = {"lying", "lie", "deceptive", "deception", "guilty", "truthful"}

    checked_values = [
        d["stress"]["label"],
        d["transcript"],
        str(d["baseline_deviation"]),
    ]
    for value in checked_values:
        lowered = str(value).lower()
        for word in banned:
            assert word not in lowered, f"banned word {word!r} leaked into {value!r}"
