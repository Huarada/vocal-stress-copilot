"""Detects an answer that isn't one.

Real failure, 2026-09-06: asked to explain a turn, the agent replied in full with

    "To explain the high arousal signal in turn t0008, I need to retrieve the specific
     acoustic evidence associated with that segment. Let me call `get_turn_evidence`
     for turn t0008."

...and stopped. The reviewer got a narrated intention instead of an explanation. The
root cause was a prompt bug (fixed separately: the prompt told it to call tools that
don't exist in context mode), but a small model will find other ways to stall, so the
symptom is worth catching independently of any one cause.

Pure functions, no I/O.
"""
from __future__ import annotations

import re

_TOOL_MENTIONS = (
    "get_turn_evidence",
    "compare_to_baseline",
    "list_flagged_turns",
    "get_transcript",
    "tool call",
    "function call",
)

_INTENT_PHRASES = (
    r"\blet me\b",
    r"\blet's\b",
    r"\bi will\b",
    r"\bi'?ll\b",
    r"\bi need to\b",
    r"\bi'?m going to\b",
    r"\bi am going to\b",
    r"\bi should\b",
    r"\bfirst,? i\b",
)

# An answer this short that is still announcing what it's about to do has not answered.
# A long answer that merely mentions a tool in passing is fine.
_STALL_LENGTH_CEILING = 600

CONTINUATION_INSTRUCTION = (
    "You did not actually answer — you described what you were about to do. There are no "
    "tools to call and nothing further will be retrieved for you: the complete session "
    "evidence is already in your context. Answer the question now, directly and fully, "
    "using that evidence."
)


def is_stalled_announcement(text: str) -> bool:
    """True when the reply announces an intention to fetch/inspect something rather than
    answering. Requires all three signals — brevity, a retrieval reference, and a
    first-person intent phrase — so a genuine short answer that happens to name a tool
    isn't misread as a stall."""
    if not text or len(text) > _STALL_LENGTH_CEILING:
        return False

    lowered = text.lower()
    mentions_retrieval = any(mention in lowered for mention in _TOOL_MENTIONS)
    if not mentions_retrieval:
        return False

    return any(re.search(phrase, lowered) for phrase in _INTENT_PHRASES)
