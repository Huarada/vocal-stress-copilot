"""ADR-031: code-level enforcement of the no-deception-vocabulary rule on the Analyst
Agent's generated text, after a real live answer violated the prompt-level rule."""
import httpx
import pytest

from voicestress.application.vocabulary_guard import (
    SAFE_FALLBACK_ANSWER,
    contains_violation,
    find_banned_terms,
)
from voicestress.infrastructure.agents.llm_gateway_client import AnalystAgent


# --- detection ---------------------------------------------------------------------


def test_detects_the_real_violating_sentence_from_the_live_run():
    """Verbatim from the 2026-09-06 live answer that motivated this guard."""
    text = "the candidate was likely suppressing their initial statement (the 'lie')"
    assert contains_violation(text)
    assert "lie" in find_banned_terms(text)


def test_hedged_mention_of_deception_is_allowed():
    """From the same live answer as the test above — but unlike the "suppressing their
    lie" sentence, this one *refuses* to claim deception. The first guard version
    flagged it too, which was wrong: negating the claim is the behaviour we want."""
    text = "we cannot rely on jitter or shimmer metrics to detect deception here"
    assert not contains_violation(text)


@pytest.mark.parametrize(
    "text",
    [
        "The speaker was lying about the timeline.",
        "This looks deceptive to me.",
        "They lied on turn 4.",
        "The candidate seems guilty.",
    ],
)
def test_detects_banned_terms(text):
    assert contains_violation(text)


def test_clean_answer_passes():
    text = (
        "Turn 7 showed elevated arousal, about 2 standard deviations above this "
        "speaker's own baseline. Pitch variability rose and there was a longer pause "
        "before the answer began."
    )
    assert not contains_violation(text)


def test_word_boundary_prevents_false_positives():
    """'believe' contains 'lie'; 'relief' contains 'lie'. Substring matching would
    make the guard fire on ordinary language and render it useless."""
    text = "I believe the relief in their voice was audible, and the client applied."
    assert not contains_violation(text)
    assert find_banned_terms(text) == []


def test_standard_disclaimer_is_not_a_violation():
    """The agent is *required* to say this — the guard must not fire on the sentence
    that states the very rule it enforces."""
    text = (
        "Turn 3 showed elevated arousal. Disclaimer: this is a vocal stress signal "
        "relative to the speaker's own baseline, not a determination of truthfulness."
    )
    assert not contains_violation(text)


def test_disclaimer_exemption_does_not_whitewash_a_real_violation():
    text = (
        "The candidate was lying here. Not a determination of truthfulness, of course."
    )
    assert contains_violation(text)


# --- assertion vs. refusal (the false-positive fix) --------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # These are the agent doing its job — refusing, hedging, stating the rule.
        # The first guard version flagged all of them, which made it refuse good answers.
        "I cannot determine whether the speaker was lying.",
        "This does not indicate deception.",
        "I can't tell you if they were being truthful — the system measures no evidence for that.",
        "This is not an indicator of lying, only of vocal arousal.",
        "We cannot rely on jitter or shimmer to detect deception here.",
        "The system is not designed to detect deception, only arousal deviation.",
        "There is no evidence of deception in an acoustic measurement.",
        "I measure vocal arousal rather than truthfulness.",
    ],
)
def test_refusals_and_disclaimers_are_not_violations(text):
    assert not contains_violation(text), f"false positive on a legitimate refusal: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        # These ASSERT something about honesty — the actual harm, flagged even when a
        # negation word is present ("not truthful" is a deception claim, not a hedge).
        "The candidate was not truthful about the timeline.",
        "He wasn't being honest in that answer.",
        "The speaker was lying.",
        "This indicates deception.",
        "The elevated pitch suggests lying.",
        "She appears deceptive on turn four.",
        "The candidate was likely suppressing their initial statement (the 'lie').",
        # Claiming someone WAS honest is equally a truthfulness determination.
        "The candidate was truthful here.",
        # Verbatim from a real answer: contrasting the signal against "a truthful
        # response" characterises the speaker's answer as untruthful. A blanket
        # "rather than" refusal marker let this through.
        "For turn 4, the model identified an acoustic stress signal rather than a truthful response.",
        "This reflects arousal instead of an honest answer.",
    ],
)
def test_assertions_about_honesty_are_violations(text):
    assert contains_violation(text), f"missed a real assertion: {text!r}"


def test_mixed_answer_with_one_bad_sentence_is_flagged():
    """A long, mostly-good answer with one assertive sentence buried in it must still
    be caught — the check is per sentence, not per answer."""
    text = (
        "Turn 5 showed elevated arousal, about 2 SD above baseline. "
        "The model attended to the 2-3 kHz band early in the clip. "
        "In my view the candidate was lying about the deadline. "
        "This should inform rather than replace your judgement."
    )
    assert contains_violation(text)


# --- AnalystAgent integration -------------------------------------------------------


def _agent_with_scripted_answers(answers: list[str]) -> AnalystAgent:
    remaining = list(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        content = remaining.pop(0)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
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


def test_clean_first_answer_returned_untouched():
    agent = _agent_with_scripted_answers(["Elevated arousal, 2 SD above baseline."])
    assert agent.ask("why?") == "Elevated arousal, 2 SD above baseline."
    assert agent.vocabulary_violations == 0


def test_violating_answer_triggers_one_correction_attempt():
    agent = _agent_with_scripted_answers(
        [
            "The candidate was lying.",  # violates
            "Turn 7 showed elevated arousal above baseline.",  # corrected
        ]
    )
    answer = agent.ask("why?")
    assert answer == "Turn 7 showed elevated arousal above baseline."
    assert agent.vocabulary_violations == 1


def test_persistent_violation_falls_back_to_safe_answer():
    """Two strikes and the answer is refused outright — for a product whose pitch is
    'it refuses to claim deception', rendering a violating answer is worse than
    rendering none."""
    agent = _agent_with_scripted_answers(
        ["They were lying.", "Still, they were being deceptive."]
    )
    answer = agent.ask("why?")
    assert answer == SAFE_FALLBACK_ANSWER
    assert not contains_violation(answer)
    assert agent.vocabulary_violations == 2
