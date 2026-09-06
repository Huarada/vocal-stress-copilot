# CLAUDE.md — project guardrails (always loaded)

This file loads automatically into every session in this directory. It exists so the
discipline below survives context resets — read it as binding, not as background.

## Read before non-trivial work

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the *what*: system design, data flow, real
  results, implementation status.
- **[ADR.md](ADR.md)** — the *why*: every architectural decision, its trade-offs, and the
  full bug catalogue with the test that guards each one.
- **[SKILLS.md](SKILLS.md)** — the *how*: skill priorities, hard rules, settled facts,
  running instructions. **Load this explicitly at the start of any work session on this
  codebase** — don't rely on memory of a prior conversation's summary of it.

If a fact in conversation contradicts one of these three files, the file wins — update the
file (with reasoning) rather than silently acting on the newer claim.

## The four non-negotiables (SKILLS.md §2 has the full list)

1. **Never claim deception.** No output, prompt, variable, UI string, or slide says
   "lying", "deceptive", "guilty", "truthful". Vocabulary is *arousal*, *deviation from
   baseline*, *confidence*. This is enforced by
   `test_to_evidence_dict_never_uses_deception_vocabulary` — if a change would make that
   test fail, the change is wrong, not the test.
2. **The Interview Agent never receives a stress score.** No path from evidence back into
   the conversation (ADR-002). Guarded by
   `test_voice_agent_session_has_no_stress_score_input_path`.
3. **Never commit datasets, derived spectrograms, or trained weights** (ADR-016). Check
   `.gitignore` covers a new artifact path before it's ever written, not after.
4. **Group by speaker in every split, always** (ADR-011). An ungrouped split produces a
   beautiful, meaningless number.

## Working discipline

- **Every architectural decision gets an ADR.** New entry in [ADR.md](ADR.md) — Context,
  Decision, Consequences, with the cost stated explicitly. Append; don't rewrite history.
  If a decision is reversed, add a new ADR marked as superseding the old one rather than
  editing it away — the wrong turn is often the most useful part of the record.
- **Verify artifacts, don't trust names.** Most bugs in Appendix A of ADR.md came from
  trusting a layer name, a field name, a doc's prose, or a `.summary()` line instead of
  loading the checkpoint / connecting to the server / printing the raw event. When a fact
  is checkable, check it before building on it.
- **A bare `except: pass` (or `except (Types): pass`) is a defect on sight, not a style
  preference** (ADR-027). It took three separate silent live-session failures — each
  with a different root cause — to learn this the hard way. Every exception branch in
  a long-running loop prints something identifying what happened, even when the branch
  is otherwise a no-op.
- **Mark what's unverified, inline.** An `# ASSUMPTION:` comment at the exact line that
  depends on an unconfirmed fact is what turned two real live-run bugs into
  minutes-long fixes instead of hours. Write the comment before the bug, not after.
- **The concept → implementation → validation loop is not optional.** A claim about
  accuracy, protocol behavior, or correctness is not settled until it's been run against
  real data or a real connection and the result written down — including when the result
  is disappointing. Report the honest number; don't tune the threshold to make a run pass.
- **New tests for new bugs.** Every bug that reaches running code gets a regression test
  before the fix is considered done, and a row in ADR.md's Appendix A.
- **A guard is not accepted until it has been made to fail** (ADR-039). Passing is not
  evidence that a test works — break the invariant it claims to protect, confirm *that
  test* goes red, restore, and verify the file is byte-identical. One of eight frontend
  guards was written green-and-useless and only mutation caught it. This applies to
  guards protecting a catalogued bug, not to every assertion in the suite.

## Scope reminder

Hackathon: "Build Voice AI Agents on AssemblyAI" (lablab.ai × AssemblyAI), submission
deadline **Sep 30, 2026, 12:00 BRT**. Portuguese for conversation with Lucas; English for
repo artifacts, code, commit messages, and anything a judge might read.
