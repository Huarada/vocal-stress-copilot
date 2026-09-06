# Practice script — exercising the live agent's mechanics

For `python scripts/run_interview.py`. Each block tests one specific thing this project
has already found evidence about (ADR-025, ADR-026, ADR-021, ADR-022) — read the "tests"
line before each block so a result is interpretable, not just noise.

**Ground rule for every line below: speak it at a normal, unhurried pace — don't rush.**
Anything under ~1 second of actual speech gets flagged `score_reliable: false`
(ADR-025) and isn't a fair test of the model; it's a test of the padding artifact.
After the session, check `artifacts/sessions/<latest>.json` for `stress.duration_ms` and
`stress.score_reliable` on each turn before reading anything into the score itself.

---

## Block 1 — Baseline (say these first, calm/neutral delivery)

The Interview Agent asks similar opening questions itself; these are what to answer with.

1. "Hi there, my name is Lucas, thanks for having me today."
2. "I'm doing well, the connection sounds clear on my end."
3. "I've been working in software development for a few years now, mostly backend systems."

## Block 2 — Consistency, duration controlled

**Tests**: on the previous session, the identical short phrase "Yeah, that's for sure."
(400ms) scored 0.95 / 0.67 / 0.81 across three repeats — real instability, but confounded
with the short-clip padding artifact. Say the same longer sentence three times, same
neutral tone each time, and see whether the score is now noticeably more stable once
duration is controlled for.

4. "Yes, I'm absolutely certain that this is correct, no doubts at all."
5. *(repeat line 4 exactly)*
6. *(repeat line 4 exactly, one more time)*

## Block 3 — Vocal energy, content held constant

**Tests**: whether the model responds to genuine vocal energy/arousal when the words
don't change — this is the axis it was actually trained to read (RAVDESS's acted
emotions), as opposed to semantic confidence/hedging language, which it never sees at
all (it has no access to text, only the spectrogram).

7. Flat, quiet, unhurried: "I think everything is going fine so far, nothing unusual to
   report."
8. Same words, loud and animated, real emphasis on "fine" and "nothing": "I think
   everything is going FINE so far, NOTHING unusual to report!"

## Block 4 — True vs. false content, tone and duration held constant

**Tests**: the honest expectation, restated — this should NOT show a clean, consistent
split. If it does, be suspicious of the result, not impressed by it (ARCHITECTURE.md §1's
whole premise is that this signal is not a truth detector). Say both lines in the *same*
calm, matter-of-fact tone, similar pace, so tone isn't a confound this time.

9. (true) "Two plus two equals four, that's a basic fact."
10. (false) "Two plus two equals five, that's a basic fact."
11. (true or false, your choice) "I woke up this morning around seven o'clock and had
    breakfast before starting work."

## Block 5 — Hedging speech, but long enough to isolate content from duration

**Tests**: on the previous session, "I really don't know what to say." (2000ms — already
above the reliability threshold) scored 0.03, genuinely low, not a duration artifact.
This checks whether that specific pattern — hedging language delivered quietly/flatly —
replicates as a real (if modest) low-arousal signal now that we know to distinguish it
from the short-clip noise.

12. "Um, I'm not entirely sure how to answer that, let me think about it for a moment."
13. "I think... I might be wrong about this, but I'll give it my best guess."

---

## After the session

1. Open the saved `artifacts/sessions/*.json` and check, per turn: `duration_ms`,
   `score_reliable`, `stress.score`, and `prosody_raw.f0_detected`.
2. Discard (don't interpret) any turn with `score_reliable: false` or
   `f0_detected: false` — those are exactly the cases ADR-025/ADR-026 exist for.
3. Compare Block 3's pair (7 vs 8) — a real, interpretable gap here (flat → higher score
   with the same words) is the strongest sanity check available that the model reads
   vocal energy, since content is literally identical.
4. Compare Block 4's pair (9 vs 10) — expect noise, not a clean split. A messy result
   here is the system working as designed, not a failure to report.
