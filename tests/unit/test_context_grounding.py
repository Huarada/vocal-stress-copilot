"""Covers ADR-030's context-grounding fallback: the account's only reachable model
(`qwen3.5-4b-32k-fast`) rejects `tools` with HTTP 400, so the Analyst Agent must be able
to ground on pre-rendered session evidence instead of on-demand tool calls.
"""
import json

import httpx
import pytest

from voicestress.domain.entities import InterviewSession, TurnEvidence
from voicestress.domain.value_objects import ArousalPrediction, ProsodyFeatures, SpeakerId
from voicestress.infrastructure.agents.llm_gateway_client import AnalystAgent
from voicestress.infrastructure.agents.tool_definitions import render_session_evidence


def _turn(turn_id="t0001", prob=0.9, t_start=0, t_end=2000, f0_detected=True, baseline_dev=None):
    return TurnEvidence(
        turn_id=turn_id,
        session_id="s1",
        t_start_ms=t_start,
        t_end_ms=t_end,
        is_baseline_turn=False,
        transcript="I think that's right.",
        arousal=ArousalPrediction(probability=prob, model_version="v1"),
        prosody_raw=ProsodyFeatures(
            f0_mean_hz=180.0, f0_std_hz=20.0, jitter_local_pct=1.0, shimmer_local_pct=5.0,
            hnr_db=15.0, speech_rate_syll_s=4.0, f0_detected=f0_detected,
            asr_mean_confidence=0.88,
        ),
        baseline_deviation=baseline_dev,
    )


def _session(turns) -> InterviewSession:
    session = InterviewSession(session_id="s1", speaker=SpeakerId("dashboard", "c1"))
    for t in turns:
        session.add_turn(t)
    return session


# --- render_session_evidence ------------------------------------------------------


def test_rendered_evidence_includes_transcript_and_score():
    text = render_session_evidence(_session([_turn()]))
    assert "t0001" in text
    assert "I think that's right." in text
    assert "0.900" in text


def test_rendered_evidence_flags_unreliable_short_turn():
    """A short turn must carry its reliability warning into the context, since in this
    mode the prompt's 'check score_reliable first' instruction has nothing else to read."""
    text = render_session_evidence(_session([_turn(t_start=0, t_end=300)]))
    assert "score_reliable: False" in text
    assert "low-confidence" in text


def test_rendered_evidence_marks_pitch_not_detected():
    text = render_session_evidence(_session([_turn(f0_detected=False)]))
    assert "pitch detection: FAILED" in text
    assert "placeholder, not a measurement" in text


def test_rendered_evidence_states_pitch_success_positively():
    """The model invented a "pitch was not detected" limitation for a turn whose pitch
    WAS measured (113.8 Hz), because availability was only implied by the presence of
    numbers. State it outright — implicit presence invites invention just as silence
    does (same principle as the Grad-CAM NOT AVAILABLE line)."""
    text = render_session_evidence(_session([_turn(f0_detected=True)]))
    assert "pitch detection: SUCCEEDED" in text
    assert "real measurements" in text


def test_rendered_evidence_states_gradcam_absence_explicitly():
    """A session predating Grad-CAM summaries produced an invented claim about "the
    low-frequency range" when the line was simply omitted."""
    turn = _turn()
    turn.gradcam_description = None
    text = render_session_evidence(_session([turn]))
    assert "NOT AVAILABLE" in text
    assert "do not describe or guess it" in text


def test_rendered_evidence_includes_gradcam_description_when_present():
    turn = _turn()
    turn.gradcam_description = "attention concentrated around 376 Hz, early in the clip"
    text = render_session_evidence(_session([turn]))
    assert "376 Hz" in text


def test_rendered_evidence_says_baseline_unavailable_when_missing():
    text = render_session_evidence(_session([_turn(baseline_dev=None)]))
    assert "baseline_deviation: unavailable" in text


def test_rendered_evidence_includes_baseline_deviation_when_present():
    text = render_session_evidence(_session([_turn(baseline_dev={"f0_mean_z": 2.1})]))
    assert "f0_mean_z=+2.10" in text


def test_pitch_derived_z_scores_are_withheld_when_pitch_failed():
    """The context used to say "these z-scores are placeholders, don't describe them as
    real" and then list them anyway. The model resolved that contradiction by inventing
    a specific pitch value ("f0_mean = 117.4 Hz") for a turn with no pitch data at all.
    Withhold rather than disclaim."""
    text = render_session_evidence(
        _session([
            _turn(
                f0_detected=False,
                baseline_dev={"f0_mean_z": -1.23, "shimmer_z": -1.34, "speech_rate_z": 0.18},
            )
        ])
    )
    assert "f0_mean_z" not in text
    assert "shimmer_z" not in text
    assert "speech_rate_z=+0.18" in text  # pitch-independent, so it survives
    assert "pitch-derived z-scores withheld" in text


def test_all_z_scores_withheld_when_none_survive_pitch_failure():
    text = render_session_evidence(
        _session([_turn(f0_detected=False, baseline_dev={"f0_mean_z": -1.23})])
    )
    assert "baseline_deviation: withheld" in text
    assert "-1.23" not in text


def test_rendered_evidence_explains_how_to_read_z_scores():
    """The model reported "+0.18" as "1.8 times higher than average". A z-score is not
    a ratio; nothing in the context said so. An unexplained unit is another gap."""
    text = render_session_evidence(_session([_turn()]))
    assert "HOW TO READ THESE NUMBERS" in text
    assert "standard deviations" in text
    assert "not ratios or percentages" in text


def test_rendered_evidence_closes_with_scope_restriction():
    text = render_session_evidence(_session([_turn()]))
    assert "ONLY discuss the turns and numbers above" in text


# --- AnalystAgent grounding modes -------------------------------------------------


def _agent(mode, context=None, capture=None):
    def handle(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )

    return AnalystAgent(
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        api_key="k",
        system_prompt="be careful",
        tool_schemas=[{"name": "get_turn_evidence", "parameters": {}}],
        tool_handlers={},
        grounding_mode=mode,
        grounding_context=context,
    )


def test_context_mode_never_sends_tools_in_payload():
    """The whole point: this account's model returns HTTP 400 if `tools` is present."""
    sent = []
    agent = _agent("context", context="SESSION s1 — evidence here", capture=sent)
    agent.ask("why?")
    assert "tools" not in sent[0]


def test_tools_mode_still_sends_tools_in_payload():
    sent = []
    agent = _agent("tools", capture=sent)
    agent.ask("why?")
    assert "tools" in sent[0]


def test_context_mode_injects_evidence_into_system_message():
    agent = _agent("context", context="SESSION s1 — turn t0001 scored 0.9")
    system_message = agent.messages[0]["content"]
    assert "be careful" in system_message  # original prompt preserved
    assert "turn t0001 scored 0.9" in system_message
    assert "the only data you may discuss" in system_message


def test_context_mode_requires_context():
    with pytest.raises(ValueError, match="requires grounding_context"):
        _agent("context", context=None)


def test_invalid_grounding_mode_rejected():
    with pytest.raises(ValueError, match="must be 'tools' or 'context'"):
        _agent("magic")


# --- session-level facts (ADR-037) ------------------------------------------------
# Both failures below were produced by the live agent against session s_1788556268
# after being asked the dashboard's own starter prompts.


def _scored(turn_id, prob, t_end=2000, f0_detected=True):
    return _turn(turn_id=turn_id, prob=prob, t_end=t_end, f0_detected=f0_detected)


def _baseline(turn_id, prob, t_end=2000, f0_detected=True):
    t = _turn(turn_id=turn_id, prob=prob, t_end=t_end, f0_detected=f0_detected)
    t.is_baseline_turn = True
    return t


def _real_session_shape() -> InterviewSession:
    """The 14-turn session that produced both inventions, with the durations and
    pitch-tracking outcomes of the real recording — the counts asserted below are the
    live session's own, so a drifting fixture fails rather than quietly agreeing."""
    return _session([
        _baseline("t0001", 0.640, t_end=4600), _baseline("t0002", 0.747, t_end=3200),
        _baseline("t0003", 0.751, t_end=500, f0_detected=False),
        _scored("t0004", 0.997, t_end=2400),
        _scored("t0005", 0.556, t_end=700, f0_detected=False),
        _scored("t0006", 0.838, t_end=2200),
        _scored("t0007", 0.958, t_end=400, f0_detected=False),
        _scored("t0008", 0.855, t_end=2200),
        _scored("t0009", 0.866, t_end=300, f0_detected=False),
        _scored("t0010", 0.001, t_end=2600, f0_detected=False),
        _scored("t0011", 0.001, t_end=2800),
        _scored("t0012", 0.000, t_end=1100, f0_detected=False),
        _scored("t0013", 0.001, t_end=2500, f0_detected=False),
        _scored("t0014", 0.007, t_end=2000, f0_detected=False),
    ])


def test_summary_names_the_true_maximum():
    """Asked "what drove the highest score?", the agent answered 0.958 at t0007. The
    real maximum was 0.997 at t0004 — it had ranked fourteen rows of prose by eye."""
    block = render_session_evidence(_real_session_shape())
    head = block[: block.index("--- t0001")]
    assert "t0004 at 0.997" in head
    assert "0.958" not in head, "a lower score is presented alongside the maximum"


def test_summary_enumerates_the_only_valid_turn_ids():
    """Asked which turn stood out, the agent cited "t0039 vs t0040" in a session whose
    ids run t0001-t0014. An unenumerated id space is a gap, and gaps get invented into."""
    block = render_session_evidence(_real_session_shape())
    head = block[: block.index("--- t0001")]
    for n in range(1, 15):
        assert f"t{n:04d}" in head
    assert "ONLY ids that exist" in head
    assert "Never cite a turn id outside that list" in head


def test_summary_separates_an_unreliable_overall_maximum():
    """When the loudest number in the session is also the least trustworthy, saying only
    "highest: 0.99" would launder a 400ms blip into the headline finding."""
    block = render_session_evidence(
        _session([_scored("t1", 0.60), _scored("t2", 0.99, t_end=400)])
    )
    head = block[: block.index("--- t1")]
    assert "highest arousal among reliable scored turns: t1 at 0.600" in head
    assert "t2 at 0.990" in head and "score_reliable: False" in head
    assert "low-confidence caveat" in head


def test_summary_refuses_to_rank_when_no_scored_turn_is_reliable():
    block = render_session_evidence(
        _session([_scored("t1", 0.9, t_end=400), _scored("t2", 0.8, t_end=300)])
    )
    head = block[: block.index("--- t1")]
    assert "no scored turn met the reliability floor" in head
    assert "supports no ranking by score" in head


def test_summary_counts_reliability_pitch_and_composition():
    block = render_session_evidence(_real_session_shape())
    head = block[: block.index("--- t0001")]
    assert "10 of 14 turns met the duration floor" in head
    assert "succeeded on 6 of 14 turns" in head
    assert "3 calibration/baseline turns, 11 scored turns" in head


def test_summary_names_the_unreliable_and_pitchless_turns_not_just_their_count():
    """Given counts alone the agent still grouped by eye, calling t0010 (2600ms,
    reliable) low-confidence. The membership of each set is stated, not implied."""
    head = render_session_evidence(_real_session_shape())
    head = head[: head.index("--- t0001")]
    assert "The ONLY turns that did not: t0003, t0005, t0007, t0009" in head
    assert "t0010" not in head.split("The ONLY turns that did not:")[1].split(".")[0]
    assert (
        "The ONLY turns where it failed: t0003, t0005, t0007, t0009, t0010, t0012, "
        "t0013, t0014" in head
    )


def test_summary_survives_a_session_with_only_baseline_turns():
    block = render_session_evidence(_session([_baseline("t1", 0.5)]))
    assert "no scored turn met the reliability floor" in block
