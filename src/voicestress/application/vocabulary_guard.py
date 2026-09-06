"""Code-level enforcement of ADR-001's vocabulary rule on the Analyst Agent's *output*.

ADR-017 put this rule in the system prompt and guarded the evidence *data* with a test,
while explicitly recording that the agent's natural-language phrasing "remains
prompt-level. The strongest guarantee stops at the tool boundary." On 2026-09-06 that
limitation stopped being theoretical: asked why a turn scored high, the live agent
(`qwen3.5-4b-32k-fast` — the only model this account can reach, ADR-024/ADR-030) wrote
*"the candidate was likely suppressing their initial statement (the 'lie')"* and *"we
cannot rely on jitter or shimmer to detect deception here"*, despite a prompt that
forbids exactly that. A 4B model does not reliably honour a negative instruction.

For a product whose entire pitch is "it refuses to claim deception", shipping an answer
that speculates about lying is worse than shipping no answer. So the rule moves into
code: detect, retry once with a correction, and fall back to a safe response rather than
render a violating answer.

Pure functions, no I/O — testable without a model.
"""
from __future__ import annotations

import re

from voicestress.domain.value_objects import BANNED_DECEPTION_TERMS

CORRECTION_INSTRUCTION = (
    "Your previous answer used vocabulary this system forbids "
    "(words like 'lie', 'lying', 'deception', 'guilty', 'truthful') or speculated about "
    "whether the speaker was being honest. This system measures vocal arousal relative "
    "to a speaker's own baseline and CANNOT determine truthfulness. Rewrite your answer "
    "using only: 'elevated arousal', 'deviation from baseline', 'acoustic stress "
    "signal', 'confidence'. Describe what the acoustic measurements show; never why the "
    "speaker's voice changed, and never whether they were being honest."
)

SAFE_FALLBACK_ANSWER = (
    "I can't answer that within this system's constraints. The question, or my attempt "
    "to answer it, moves toward a determination about whether the speaker was being "
    "honest — which this system measures no evidence for and is designed never to "
    "claim. What I can tell you is what the acoustic measurements show for a given "
    "turn: the arousal score, how reliable that score is, and how it deviates from this "
    "speaker's own baseline. Ask me about a specific turn's measurements and I'll walk "
    "you through them."
)


def find_banned_terms(text: str) -> list[str]:
    """Returns the forbidden terms present in `text`, matched on word boundaries so
    substrings inside legitimate words don't false-positive (e.g. 'believe' contains
    'lie', 'guilty' must not match inside a longer token). Case-insensitive.

    Deliberately does NOT scan for the disclaimer's own wording — the standard
    disclaimer legitimately contains "truthfulness" while explaining what the system
    does not do, so callers strip or exempt it before checking. See
    `contains_violation`.
    """
    lowered = text.lower()
    return sorted(
        term for term in BANNED_DECEPTION_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)
    )


# The distinction that matters is WHERE a negation attaches.
#
# Negating the *honesty itself* ("was not truthful", "wasn't being honest") asserts
# deception — a negation wearing a disclaimer's clothes. Checked first, and it wins.
_DIRECT_DENIAL_OF_HONESTY = (
    r"\b(?:not|isn'?t|wasn'?t|weren'?t|aren'?t|never)\s+(?:\w+\s+){0,2}?(?:truthful|honest)\b",
)

# Negating the *claim* ("cannot determine whether they were lying", "no evidence of
# deception") is the agent doing exactly what it's told to do. The first version of this
# guard had no such notion and refused these, which made it block good answers — worse
# for the product than the problem it was added to solve.
_EPISTEMIC_REFUSAL_MARKERS = (
    r"\bcan(?:no|')?t\b",
    r"\bcannot\b",
    r"\bdo(?:es)?\s+not\b",
    r"\bdo(?:es)?n'?t\b",
    r"\bdid\s*n[o']?t\b",
    r"\bunable\b",
    r"\bnever\b",
    r"\bno evidence\b",
    r"\bnot a determination\b",
    r"\bnot an indicator\b",
    r"\bnot designed\b",
    r"\bnot scientifically\b",
    # NARROWED 2026-09-06: a blanket "rather than" let through *"the model identified an
    # acoustic stress signal rather than a truthful response"* — which characterises the
    # speaker's answer as untruthful, the exact harm this guard exists for. The
    # legitimate use contrasts what the SYSTEM measures ("arousal rather than
    # truthfulness"), so only the bare abstract nouns qualify, not "a truthful <thing>".
    r"\brather than\s+(?:a\s+)?(?:determination|truthfulness|honesty|deception|lying|dishonesty)\b",
    r"\binstead of\s+(?:a\s+)?(?:determination|truthfulness|honesty|deception|lying|dishonesty)\b",
    r"\bonly of\b",
    r"\bnot\b",
)

# Constructions that assert something about the speaker's honesty with no negation at
# all — checked last, once refusals have been excluded.
_ASSERTIVE_DECEPTION_PATTERNS = (
    r"\b(?:was|were|is|are|seems?|appears?|sounds?|looks?|being)\s+(?:\w+\s+){0,2}?"
    r"(?:lying|deceptive|dishonest|untruthful|truthful|honest)\b",
    r"\bsuppress\w*\b[^.!?]{0,40}\b(?:lie|lies|truth)\b",
    r"\b(?:indicat\w+|suggest\w+|shows?|reveal\w+|evidence of|sign of|proof of)\s+"
    r"(?:\w+\s+){0,2}?(?:deception|lying|dishonesty)\b",
)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;:])\s+|\n+", text) if s.strip()]


def contains_violation(text: str, allow_disclaimer: bool = True) -> bool:
    """True if `text` *asserts* something about the speaker's honesty.

    Sentence-level rather than whole-text, and assertion-aware rather than plain term
    matching, because the first version was too blunt to ship: it flagged the agent's
    own required disclaimers ("I cannot determine whether they were lying"), refusing
    perfectly good answers. Blocking a correct explanation defeats the XAI purpose as
    surely as rendering a harmful one does.

    Three ordered passes per sentence, on the principle that what matters is where a
    negation attaches:
      1. Negating the honesty itself ("was not truthful") asserts deception → violation,
         even though a negation is present.
      2. Otherwise, negating the *claim* ("cannot determine whether they were lying",
         "no evidence of deception") is the agent following its instructions → allowed.
      3. Otherwise, an assertive construction ("was lying", "indicates deception",
         "suppressing the lie") or a bare banned term → violation.
    """
    for sentence in _sentences(text):
        lowered = sentence.lower()

        if any(re.search(p, lowered) for p in _DIRECT_DENIAL_OF_HONESTY):
            return True

        if not find_banned_terms(sentence) and not any(
            re.search(p, lowered) for p in _ASSERTIVE_DECEPTION_PATTERNS
        ):
            continue

        if allow_disclaimer and any(
            re.search(m, lowered) for m in _EPISTEMIC_REFUSAL_MARKERS
        ):
            continue

        return True

    return False
