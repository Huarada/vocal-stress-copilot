"""JSON-Schema tool definitions for the Analyst Agent (ARCHITECTURE.md §5) plus the
handlers that actually resolve them against an `InterviewSession`.

Everything here is pure Python — no network — so the "does the LLM's tool call return
the right grounded data" question is testable without a live LLM Gateway connection.
The Analyst Agent's system prompt (agents/prompts/analyst.md) forbids asserting anything
these tools didn't return; this module is what makes that enforceable rather than
aspirational, since it's the only source of truth the agent is ever given.
"""
from __future__ import annotations

from typing import Any, Callable

from voicestress.domain.entities import InterviewSession

# --- JSON Schema tool definitions, per the Voice Agent / LLM Gateway tool-calling spec
# (session.tools[] with type:"function", parameters as JSON Schema — ARCHITECTURE.md §5) ---

LIST_FLAGGED_TURNS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "list_flagged_turns",
    "description": (
        "List the interview turns whose vocal arousal signal deviated most from the "
        "candidate's own baseline. Returns turn ids and scores only — call "
        "get_turn_evidence for detail on any specific turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "min_score": {
                "type": "number",
                "description": "Minimum arousal probability (0-1) to include.",
                "default": 0.6,
            }
        },
        "required": [],
    },
}

GET_TURN_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "get_turn_evidence",
    "description": (
        "Get the full evidence record for one interview turn: transcript, arousal "
        "score, baseline z-deviations, raw prosodic features, and Grad-CAM/spectrogram "
        "artifact paths. This is the ONLY source of acoustic detail — never state a "
        "number that didn't come from this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {"turn_id": {"type": "string"}},
        "required": ["turn_id"],
    },
}

COMPARE_TO_BASELINE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "compare_to_baseline",
    "description": (
        "Get a turn's prosodic z-deviations against the candidate's own calibration "
        "baseline, plus how many calibration turns that baseline rests on. If the "
        "baseline was not reliable, this says so explicitly instead of returning "
        "unsupported z-scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {"turn_id": {"type": "string"}},
        "required": ["turn_id"],
    },
}

GET_TRANSCRIPT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "get_transcript",
    "description": "Get the verbatim transcript text for one turn, for quoting exactly.",
    "parameters": {
        "type": "object",
        "properties": {"turn_id": {"type": "string"}},
        "required": ["turn_id"],
    },
}

ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    LIST_FLAGGED_TURNS_SCHEMA,
    GET_TURN_EVIDENCE_SCHEMA,
    COMPARE_TO_BASELINE_SCHEMA,
    GET_TRANSCRIPT_SCHEMA,
]


class ToolExecutionError(ValueError):
    pass


def _render_session_summary(session: InterviewSession) -> list[str]:
    """Session-scope facts, precomputed and stated outright.

    Same principle as the per-turn presence/absence statements (ADR-034), applied one
    level up. Asked "what drove the highest score?", the model scanned fourteen turns in
    prose and answered 0.958 at t0007 — the true maximum was 0.997 at t0004. Asked which
    turn stood out, it named the right turn and then cited "t0039 vs t0040", neither of
    which exists in a fourteen-turn session. Ranking and counting across many rows is
    exactly what a language model should not be asked to do from prose, and an
    unenumerated id space is one more gap to fill with invention.

    So: rank here, count here, and enumerate the ids that exist.
    """
    turns = session.turns
    if not turns:
        return []

    scored = [t for t in turns if not t.is_baseline_turn]
    reliable_scored = [t for t in scored if t.arousal_score_reliable]
    by_score = lambda t: t.arousal.probability  # noqa: E731

    lines = [
        "SESSION-LEVEL FACTS (precomputed — use these instead of deriving your own; "
        "do not re-rank or re-count the turns yourself):",
        "  turn ids in this session, and the ONLY ids that exist: "
        + ", ".join(t.turn_id for t in turns),
        "  Never cite a turn id outside that list. If a question refers to one that is "
        "not listed, say it is not part of this session.",
    ]

    if reliable_scored:
        top = max(reliable_scored, key=by_score)
        lines.append(
            f"  highest arousal among reliable scored turns: {top.turn_id} at "
            f"{top.arousal.probability:.3f} — this is the answer to "
            f'"which turn stands out" and "what scored highest".'
        )
        mean = sum(t.arousal.probability for t in reliable_scored) / len(reliable_scored)
        lines.append(
            f"  mean arousal across the {len(reliable_scored)} reliable scored turns: "
            f"{mean:.3f}"
        )
    else:
        top = None
        lines.append(
            "  highest arousal among reliable scored turns: none — no scored turn met "
            "the reliability floor, so the session supports no ranking by score."
        )

    if scored:
        top_any = max(scored, key=by_score)
        if top is None or top_any.turn_id != top.turn_id:
            lines.append(
                f"  highest arousal among ALL scored turns including low-confidence "
                f"ones: {top_any.turn_id} at {top_any.arousal.probability:.3f} "
                f"(score_reliable: {top_any.arousal_score_reliable}). Report the "
                f"reliable figure above as the headline; mention this one only with "
                f"its low-confidence caveat attached."
            )

    # Naming the ids, not just counting them. With counts alone the model still had to
    # group fourteen rows by eye, and it put t0010 (2600ms, reliable) in the unreliable
    # set. Every cross-turn grouping the answer needs is cheaper to state than to derive.
    unreliable = [t.turn_id for t in turns if not t.arousal_score_reliable]
    no_pitch = [t.turn_id for t in turns if not t.prosody_raw.f0_detected]
    lines.append(
        f"  reliability: {len(turns) - len(unreliable)} of {len(turns)} turns met the "
        f"duration floor. The ONLY turns that did not: "
        + (", ".join(unreliable) if unreliable else "none")
        + ". Do not describe any other turn as low-confidence."
    )
    lines.append(
        f"  pitch tracking: succeeded on {len(turns) - len(no_pitch)} of {len(turns)} "
        f"turns. The ONLY turns where it failed: "
        + (", ".join(no_pitch) if no_pitch else "none")
        + ". Do not describe any other turn's pitch as missing."
    )
    lines.append(
        f"  composition: {len(turns) - len(scored)} calibration/baseline turns, "
        f"{len(scored)} scored turns."
    )
    return lines


def render_session_evidence(session: InterviewSession) -> str:
    """Renders every turn's evidence as a compact text block for context-injection
    grounding — the fallback for models that don't support tool calling (ADR-030).

    Same grounding guarantee as the tools (ADR-004): the Analyst Agent can only speak
    about data actually placed in front of it. The difference is *when* the data is
    selected — up front for the whole session, instead of on demand per question. For
    an interview-sized session (10-20 turns) the whole thing fits comfortably in a 32k
    context window, so nothing is lost by pre-loading it.
    """
    lines = [
        f"SESSION {session.session_id} — {len(session.turns)} turns recorded.",
        "",
        # LEGEND added 2026-09-06: the model read "speech_rate_z=+0.18" as "1.8 times "
        # higher than average". A z-score is not a ratio, and nothing in the context
        # said so. Same principle as stating presence and absence outright — an
        # unexplained unit is another gap to fill with invention.
        "HOW TO READ THESE NUMBERS:",
        "  z-scores (…_z) are standard deviations from THIS speaker's own calibration "
        "baseline, not ratios or percentages. +1.0 means one standard deviation above "
        "that speaker's own norm; +0.18 is a small deviation, not '18% higher' or "
        "'1.8 times higher'.",
        "  arousal score is a 0-1 model output, not a probability of anything about the "
        "speaker's state.",
        "",
    ]

    summary = _render_session_summary(session)
    if summary:
        lines.extend(summary)
        lines.append("")

    for turn in session.turns:
        kind = "BASELINE/calibration" if turn.is_baseline_turn else "scored"
        lines.append(f"--- {turn.turn_id} ({kind}) ---")
        lines.append(f"transcript: {turn.transcript!r}")
        lines.append(
            f"arousal score: {turn.arousal.probability:.3f} ({turn.arousal.label.value}), "
            f"duration: {turn.duration_ms}ms, "
            f"score_reliable: {turn.arousal_score_reliable}"
        )
        if not turn.arousal_score_reliable:
            lines.append(
                "  ^ NOTE: audio shorter than the reliability floor — this score is "
                "low-confidence and must be reported as such."
            )

        prosody = turn.prosody_raw
        # Availability is stated POSITIVELY as well as negatively. An earlier version
        # printed values when present and a warning when absent, leaving "these numbers
        # are real" merely implied — and the model invented the limitation anyway,
        # announcing "pitch was not detected" for a turn whose pitch was measured at
        # 113.8 Hz. Implicit presence invites invention exactly as silence does.
        if prosody.f0_detected:
            lines.append(
                "pitch detection: SUCCEEDED — the values below are real measurements."
            )
            lines.append(
                f"prosody: f0_mean={prosody.f0_mean_hz:.1f}Hz f0_std={prosody.f0_std_hz:.1f} "
                f"jitter={prosody.jitter_local_pct:.2f}% shimmer={prosody.shimmer_local_pct:.2f}% "
                f"hnr={prosody.hnr_db:.1f}dB rate={prosody.speech_rate_syll_s:.1f}/s"
            )
        else:
            lines.append(
                "pitch detection: FAILED for this turn — every pitch-derived value "
                "(f0, jitter, shimmer, HNR) is a placeholder, not a measurement, and so "
                "are the z-scores computed from them. Do not describe any of them as "
                "real. Only speech_rate is independent of pitch tracking."
            )
        if prosody.asr_mean_confidence is not None:
            lines.append(f"asr_mean_confidence: {prosody.asr_mean_confidence:.3f}")
        if prosody.onset_latency_ms is not None:
            lines.append(f"onset_latency_ms: {prosody.onset_latency_ms:.0f}")

        # ADR-032: the actual explanation of the score — which region of the
        # spectrogram the CNN attended to. Previously only a PNG path reached the
        # agent, which is why it could not explain any score.
        #
        # ABSENCE IS STATED EXPLICITLY, not by omission: a session recorded before this
        # field existed produced an answer claiming "the Grad-CAM attention focused on
        # the low-frequency range" — invented, because the line was simply missing from
        # the context. Silence invites invention; an explicit "not available" does not.
        if turn.gradcam_description:
            lines.append(f"model attention (Grad-CAM): {turn.gradcam_description}")
        else:
            lines.append(
                "model attention (Grad-CAM): NOT AVAILABLE for this turn (session "
                "recorded before attention summaries existed). You have no information "
                "about where the model looked — do not describe or guess it."
            )

        if turn.baseline_deviation:
            # When pitch tracking failed, the pitch-derived z-scores are computed from
            # placeholder zeros and are as fake as the values beneath them. Printing
            # "these are placeholders, don't describe them as real" and then listing
            # them anyway is a contradiction the model resolved by inventing — it
            # reported a specific "f0_mean = 117.4 Hz" for a turn whose evidence
            # contained no pitch value at all. Withhold them instead of disclaiming
            # them; only speech_rate survives a pitch-tracking failure.
            deviations = {
                k: v
                for k, v in turn.baseline_deviation.items()
                if turn.prosody_raw.f0_detected or k.startswith("speech_rate")
            }
            if deviations:
                rendered = " ".join(f"{k}={v:+.2f}" for k, v in deviations.items())
                suffix = (
                    ""
                    if turn.prosody_raw.f0_detected
                    else "  (pitch-derived z-scores withheld — pitch tracking failed)"
                )
                lines.append(f"baseline_deviation (z-scores): {rendered}{suffix}")
            else:
                lines.append(
                    "baseline_deviation: withheld — pitch tracking failed, so every "
                    "deviation for this turn would be computed from placeholder values."
                )
        else:
            lines.append(
                "baseline_deviation: unavailable (baseline not established or not "
                "reliable) — do not report z-scores for this turn."
            )
        lines.append("")

    lines.append(
        "You may ONLY discuss the turns and numbers above. If asked about anything not "
        "present here, say you don't have that information."
    )
    return "\n".join(lines)


def build_tool_handlers(session: InterviewSession) -> dict[str, Callable[[dict], Any]]:
    """Binds the tool schemas above to a specific InterviewSession, returning
    {tool_name: callable(arguments_dict) -> JSON-serializable result}. The Analyst Agent
    adapter dispatches an incoming `tool.call` event through this map — see
    infrastructure/agents/llm_gateway_client.py.
    """

    def list_flagged_turns(args: dict) -> dict:
        min_score = float(args.get("min_score", 0.6))
        flagged = session.flagged_turns(min_score=min_score)
        return {
            "turns": [
                {"turn_id": t.turn_id, "score": t.arousal.probability, "label": t.arousal.label.value}
                for t in flagged
            ]
        }

    def _require_turn(turn_id: str):
        turn = session.get_turn(turn_id)
        if turn is None:
            raise ToolExecutionError(f"No such turn_id in this session: {turn_id!r}")
        return turn

    def get_turn_evidence(args: dict) -> dict:
        turn = _require_turn(args["turn_id"])
        return turn.to_evidence_dict()

    def compare_to_baseline(args: dict) -> dict:
        turn = _require_turn(args["turn_id"])
        if turn.baseline_deviation is None:
            return {
                "turn_id": turn.turn_id,
                "baseline_reliable": False,
                "message": (
                    "Baseline not established or not reliable for this session — "
                    "deviations cannot be computed. Report this limitation rather than "
                    "guessing."
                ),
            }
        return {
            "turn_id": turn.turn_id,
            "baseline_reliable": True,
            "deviation": turn.baseline_deviation,
        }

    def get_transcript(args: dict) -> dict:
        turn = _require_turn(args["turn_id"])
        return {"turn_id": turn.turn_id, "transcript": turn.transcript}

    return {
        "list_flagged_turns": list_flagged_turns,
        "get_turn_evidence": get_turn_evidence,
        "compare_to_baseline": compare_to_baseline,
        "get_transcript": get_transcript,
    }
