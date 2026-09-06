# Architecture Decision Records

Every architectural decision taken on this project, why it was taken, and — explicitly —
what each one costs. Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (the *what*) and
[SKILLS.md](SKILLS.md) (the *how to work on it*). This file is the *why*.

**Format**: each record states Context (the forces), Decision (what we did), and
Consequences (what we gained AND what we gave up — an ADR without a stated cost is
marketing, not a record).

**Status legend**: `accepted` · `accepted, revised` (superseded by a later ADR but the
reasoning still matters) · `open` (decided provisionally, evidence still pending).

**Adding one**: append, don't edit history. If a decision is reversed, add a new ADR that
supersedes it and mark the old one `accepted, revised` — the wrong turn is often the most
useful part of the record.

---

## Index

| # | Decision | Status |
|---|----------|--------|
| [001](#adr-001) | Frame the product as vocal-arousal signalling, never deception detection | accepted |
| [002](#adr-002) | Two agents, with the Interview Agent structurally blind to scores | accepted |
| [003](#adr-003) | Per-speaker baseline calibration instead of absolute thresholds | accepted |
| [004](#adr-004) | Ground the Analyst Agent exclusively in tool results | accepted |
| [005](#adr-005) | DDD layering with ports and adapters | accepted |
| [006](#adr-006) | Inject transports rather than constructing them | accepted |
| [007](#adr-007) | SyncBus as the single turn-segmentation authority | accepted |
| [008](#adr-008) | The evidence record as the system's central contract | accepted |
| [009](#adr-009) | Reuse the inherited preprocessing and architecture verbatim | accepted |
| [010](#adr-010) | Treat the checkpoint warm-start as an A/B experiment, not an assumption | accepted |
| [011](#adr-011) | Speaker-grouped cross-validation, always | accepted |
| [012](#adr-012) | Cross-corpus generalization as the headline metric | accepted |
| [013](#adr-013) | RAVDESS arousal binarization, with `happy` excluded | accepted |
| [014](#adr-014) | MUStARD++ binarization with an ambiguity dead zone | accepted |
| [015](#adr-015) | Abandon institutionally gatekept deception corpora | accepted |
| [016](#adr-016) | Never commit datasets, derived artifacts, or trained weights | accepted |
| [017](#adr-017) | Enforce vocabulary discipline in code, not only in prompts | accepted |
| [018](#adr-018) | Quality gates as tests with documented, justified thresholds | accepted |
| [019](#adr-019) | The dashboard degrades gracefully without an API key | accepted |
| [020](#adr-020) | Match transplanted weights positionally by layer type | accepted |
| [021](#adr-021) | Discard sub-threshold speech segments before the pipeline sees them | accepted |
| [022](#adr-022) | Pair transcripts to turns FIFO, not by "most recent" | accepted |
| [023](#adr-023) | Source protocol defaults from docs and live probes, never intuition | accepted |
| [024](#adr-024) | Select the LLM by account entitlement probing, not roster reading | accepted |
| [025](#adr-025) | Flag short-clip arousal scores as unreliable instead of retraining | accepted |
| [026](#adr-026) | Distinguish "measured zero" from "extraction failed" in prosody | accepted |
| [027](#adr-027) | No exception in the live loop may be swallowed without printing | accepted |
| [028](#adr-028) | Amortized audio buffer + off-loop scoring, not O(n²) blocking inference | accepted |
| [029](#adr-029) | Use AssemblyAI's word-level confidence/timestamps as a second hesitation signal | accepted |
| [030](#adr-030) | Context-injection grounding, because the only reachable model can't call tools | accepted |
| [031](#adr-031) | Enforce the no-deception vocabulary rule in code, on the agent's output | accepted, revised by [033](#adr-033) |
| [032](#adr-032) | Give the Analyst Agent the Grad-CAM as facts, not as a file path | accepted |
| [033](#adr-033) | Make the vocabulary guard assertion-aware, not term-matching | accepted |
| [034](#adr-034) | Leave no gaps: mode-aware prompt, explicit presence/absence, stall guard | accepted |
| [035](#adr-035) | The dashboard reads evidence through the domain layer, not the session file | accepted |
| [036](#adr-036) | The Analyst Agent is a persistent panel, not a tab | accepted |
| [037](#adr-037) | Precompute session-scope facts; never ask the model to rank or count | accepted |
| [038](#adr-038) | Seed the chat panel with starter prompts, and treat them as shipped surface | accepted |
| [039](#adr-039) | A guard is not accepted until it has been made to fail | accepted |
| [040](#adr-040) | Wire the Interview Agent's reply back to a human, for the first time | accepted |
| [041](#adr-041) | Browser-based capture, decoupled from sounddevice, sharing the runner | accepted |
| [042](#adr-042) | Never report success without checking for it | accepted |
| [043](#adr-043) | A wrong guess must degrade, never flood | accepted |
| [044](#adr-044) | Give the Interview Agent exactly one tool, on the entitlement that actually grants it | accepted |
| [045](#adr-045) | Re-read the docs before building on a working capability; don't move the Analyst Agent to voice | accepted |
| [046](#adr-046) | Make TensorFlow an opt-in cost, not a fixed one, for the dashboard deployment | accepted |
| [047](#adr-047) | A public dashboard deploy ships with no API key by default | accepted |
| [048](#adr-048) | Confirmed live: the deployed no-key default fails safely from a real browser | accepted |

---

<a id="adr-001"></a>
## ADR-001 — Frame the product as vocal-arousal signalling, never deception detection

**Status**: accepted (2026-09-04) · the decision every other one inherits from

**Context.** The project began as "detect if the speaker is lying from voice spectra."
Voice-based deception detection is scientifically contested — voice stress analysis has a
history stretching back to 1970s-era devices that courts and the research community
generally do not accept. Beyond the science, an "AI lie detector" is close to unsellable
in the hiring market it targets: no corporate legal team signs off on a product whose core
claim invites disparate-impact litigation.

Empirical confirmation arrived later and independently: on a live run, the same false
statement ("3 + 3 is 7") spoken twice scored `1.00` and then `0.07`. The signal is real but
noisy, and it is a signal about *arousal*, not *truth*.

**Decision.** The system measures and reports **vocal arousal deviation relative to the
speaker's own baseline**. No component — model output, prompt, variable name, UI string,
or slide — asserts that anyone is lying. The vocabulary is *elevated arousal*, *deviation
from baseline*, *confidence*. The product's job is to direct a human reviewer's attention,
never to render a verdict.

**Consequences.**
- Gained: a claim the evidence actually supports; a product a compliance team could
  approve; a differentiator against the naive framing most competitors in this space use.
- Gained: the noisy live-test result became a *validation* of the framing rather than a
  failure of the product.
- **Given up**: the immediately legible pitch. "Lie detector" sells itself in one word;
  "baseline-relative vocal arousal signal for reviewer attention" needs a sentence. Every
  pitch and demo has to do that extra work.
- **Given up**: the ability to ever report accuracy against a deception ground truth, which
  is the metric a naive evaluator will ask for first.

---

<a id="adr-002"></a>
## ADR-002 — Two agents, with the Interview Agent structurally blind to scores

**Status**: accepted (2026-09-04)

**Context.** The obvious design is one agent that interviews *and* reacts to the stress
signal. Three things break simultaneously if it does: the agent's reaction contaminates the
acoustic signal it is measuring; the interaction becomes coercive toward the interviewee;
and the session stops being reproducible, since the questions asked now depend on the
model's own mid-conversation output.

**Decision.** Two agents with a one-way boundary. The **Interview Agent** (AssemblyAI Voice
Agent API) conducts the conversation and receives *nothing* derived from the acoustic
pipeline. The **Analyst Agent** (LLM Gateway) explains evidence to a human reviewer, after
or alongside the interview, and cannot speak to the interviewee. Scores flow *sideways*
into the evidence store, never back into the conversation.

This is enforced structurally, not by convention: `VoiceAgentSession` has no method or
field that accepts a score, and
`tests/unit/test_voice_agent_client.py::test_voice_agent_session_has_no_stress_score_input_path`
fails if one is ever added.

**Consequences.**
- Gained: signal integrity, reproducibility, and a defensible answer to "does the AI
  interrogate people harder when it thinks they're nervous?" — it cannot.
- Gained: a design decision judges and reviewers can recognize as deliberate rather than
  incidental.
- **Given up**: adaptive interviewing. An agent that knew a turn was flagged could ask a
  natural clarifying follow-up in the moment; ours cannot, so that judgment stays with the
  human reviewer, who may only see the flag after the call.
- **Given up**: some product magic in the demo — the most impressive-looking version of
  this system is exactly the one we refused to build.

---

<a id="adr-003"></a>
## ADR-003 — Per-speaker baseline calibration instead of absolute thresholds

**Status**: accepted (2026-09-04)

**Context.** Absolute arousal thresholds measure the wrong thing: a speaker with a
naturally tense or high-pitched voice, a heavy accent, a cheap microphone, or a noisy room
scores "elevated" permanently. That is a bias generator, not a signal.

The cross-corpus result later made the case empirically: the RAVDESS-trained model
over-predicts HIGH on MUStARD++ (recall `0.820`, precision `0.565`), because sitcom
dialogue is more animated than RAVDESS's "neutral/calm" even at low narrative arousal. A
fixed threshold does not survive a domain change; a within-speaker comparison does.

**Decision.** The interview opens with 2–3 deliberately neutral calibration turns. Each
prosodic feature accumulates a per-session mean and standard deviation; every later turn is
reported as a z-deviation from *that speaker's own* opening. `BaselineProfile.is_reliable`
refuses to produce z-scores from too few turns or a degenerate (zero-variance) window, and
the Analyst Agent reports that limitation explicitly rather than emitting unsupported
numbers.

**Consequences.**
- Gained: the narrow, defensible claim — "this answer deviated from how this same person
  sounded five minutes ago" — instead of "this person sounds tense."
- Gained: substantial mitigation of the accent/mic/room confounds that would otherwise make
  the system systematically unfair.
- **Given up**: the first ~3 turns of every interview, which produce no usable score.
- **Given up**: robustness when calibration is bad. If the opening turns aren't actually
  neutral (a candidate nervous from the first second), the baseline is skewed and
  everything after it is measured against the wrong reference — the system can detect
  *degenerate* baselines but not *misleading* ones.

---

<a id="adr-004"></a>
## ADR-004 — Ground the Analyst Agent exclusively in tool results

**Status**: accepted (2026-09-04)

**Context.** An LLM explaining a noisy classifier is the exact setup where fluent
speculation is most dangerous: it will happily narrate a confident causal story about why
someone's voice changed. That failure mode would turn this project into precisely the
pseudoscience ADR-001 exists to avoid.

**Decision.** The Analyst Agent has four tools (`list_flagged_turns`, `get_turn_evidence`,
`compare_to_baseline`, `get_transcript`) and a system prompt forbidding any claim not
traceable to a tool result it actually called. `compare_to_baseline` returns an explicit
`baseline_reliable: false` with a message when it cannot support z-scores, so "I don't
know" is a first-class, machine-provided answer rather than something the model has to
invent.

**Consequences.**
- Gained: every number the reviewer reads is traceable to a computed artifact.
- Gained: the XAI layer becomes an *interface* over the model rather than a second model
  speculating about a person.
- **Given up**: conversational richness. A grounded Analyst Agent is deliberately boring;
  it cannot offer the plausible-sounding narrative a reviewer might actually want.
- **Given up**: full enforceability. The tool boundary is hard, but "don't assert beyond
  the tool result" is still prompt-level for the *phrasing* of answers — code enforces what
  data reaches the model, not every word it writes about that data.

---

<a id="adr-005"></a>
## ADR-005 — DDD layering with ports and adapters

**Status**: accepted (2026-09-04)

**Context.** The system spans real-time audio, DSP, a Keras model, two LLM APIs, and a web
dashboard. Written directly, the acoustic model's file paths end up inside websocket
handlers and nothing is testable without a microphone and an API key.

**Decision.** Four layers: `domain/` (frozen value objects, entities, and `ports.py`
Protocols — zero framework imports), `application/` (use cases depending only on domain
ports), `infrastructure/` (concrete adapters implementing those ports), and `scripts/` as
composition roots that wire concrete adapters in. The dependency arrow points inward and
never outward.

**Consequences.**
- Gained: the domain and application layers run in milliseconds with no TensorFlow, no
  audio hardware, and no network — which is why 287 tests run in ~21 seconds.
- Gained: real swappability. Replacing the CNN with a wav2vec2 head, or AssemblyAI with
  another vendor, is a new adapter, not a rewrite.
- **Given up**: directness. A file count and an indirection layer that a single-file
  notebook would not have. For a 26-day hackathon this is a real cost paid up front.
- **Given up**: some of the payoff, honestly — the swappability argument is currently
  theoretical, since only one adapter exists per port.

---

<a id="adr-006"></a>
## ADR-006 — Inject transports rather than constructing them

**Status**: accepted (2026-09-04)

**Context.** `VoiceAgentSession` and `AnalystAgent` are the two components that talk to the
network, and were written before any API key existed. Constructing their connections
internally would have made both untestable until credentials arrived.

**Decision.** `VoiceAgentSession` takes a `Transport` Protocol (`send`/`recv`);
`AnalystAgent` takes an `httpx.Client`. Real implementations
(`WebsocketTransport`, a plain `httpx.Client`) are supplied at the composition root. Tests
supply `FakeTransport` and `httpx.MockTransport`.

**Consequences.**
- Gained: the entire protocol layer — message framing, event dispatch, the tool-calling
  loop, runaway-loop protection — was developed and tested with no key and no network.
- Gained: a single, clearly marked file (`websocket_transport.py`) as the only genuinely
  untestable-without-credentials component, which is exactly where the honesty boundary
  belongs.
- **Given up**: fakes prove protocol *handling*, never protocol *correctness*. Every one of
  the three field-shape bugs in [Appendix A](#appendix-a) passed the fake tests and failed
  against the real server. Injection buys testability, not truth.

---

<a id="adr-007"></a>
## ADR-007 — SyncBus as the single turn-segmentation authority

**Status**: accepted (2026-09-04)

**Context.** Two streams (raw audio and Voice Agent events) must agree on where a turn
starts and ends. The characteristic failure of this class of system is timing drift between
the audio being scored and the transcript being shown — and it fails silently.

**Decision.** One component owns the mapping from `input.speech.started`/`stopped` events to
audio slices and timestamps. Everything downstream consumes its output and never re-derives
a boundary. It is pure Python over numpy arrays, so it is fully testable with synthetic
audio and simulated events.

**Consequences.**
- Gained: one place to fix segmentation bugs — which is exactly where both ADR-021 and
  ADR-022 landed.
- Gained: unit-testable timing logic, with no hardware.
- **Given up**: it can only be as correct as the events feeding it. SyncBus was already
  correct when transcripts were being attached to the wrong turns (ADR-022) — centralizing
  segmentation does not centralize *pairing*, and conflating the two cost a debugging cycle.

---

<a id="adr-008"></a>
## ADR-008 — The evidence record as the system's central contract

**Status**: accepted (2026-09-04)

**Context.** The acoustic track and the agent track had to be built in parallel, and the
dashboard needed to render sessions produced by two different generators (a live interview
and an offline demo).

**Decision.** A single JSON contract (ARCHITECTURE.md §4) with a typed counterpart
(`TurnEvidence`), plus exact inverse constructors (`from_evidence_dict` /
`from_session_dict`) so a persisted session rehydrates into real domain objects. The
dashboard's chat endpoint therefore reuses `build_tool_handlers` unchanged rather than
reimplementing it against raw dicts.

**Consequences.**
- Gained: parallel development against a stable interface; one code path serving live and
  replayed sessions.
- Gained: a `disclaimer` field carried on every record, so the ADR-001 framing travels with
  the data instead of living only in the UI.
- **Given up**: rigidity. Changing the contract now means touching the writer, the reader,
  the tool handlers, the frontend, and the round-trip test — this already bit once, when the
  deviation-key naming diverged from the documented shape (Bug 2).

---

<a id="adr-009"></a>
## ADR-009 — Reuse the inherited preprocessing and architecture verbatim

**Status**: accepted (2026-09-04)

**Context.** The project inherited a working sarcasm-detection pipeline: 16 kHz mono →
narrowband spectrogram (80 ms window / 10 ms hop) → a 3-channel image (R = log-mel,
G = Δ-MFCC, B = ΔΔ-MFCC, each independently z-scored) → a 6,578-parameter ResNet-light.
The temptation was to "clean it up" while porting.

**Decision.** Port it as-is, including the contrast guard that rejects flat spectrograms
(the original's *"tapete cinza"* check). Change the label, not the network.

**Consequences.**
- Gained: the highest-risk component was already validated; a week of the 26-day budget went
  to integration instead of modelling.
- Gained: the 80 ms window genuinely delivers the harmonic resolution the project's original
  premise wanted — that part of the inherited work was right.
- **Given up**: modern accuracy. Self-supervised speech representations (wav2vec2, HuBERT,
  WavLM) are the 2024–2026 state of the art for this task; a spectrogram-image CNN is
  roughly 2017-era methodology. We are knowingly not competitive on raw accuracy.
- **Given up**: the chance to discover that a cleaner reimplementation would have been
  better — a rewrite would have been a new source of preprocessing drift, so we never tested
  that hypothesis.

---

<a id="adr-010"></a>
## ADR-010 — Treat the checkpoint warm-start as an A/B experiment, not an assumption

**Status**: accepted (2026-09-04)

**Context.** The plan assumed `resnet1_prelayer1.keras` was a spectrogram-domain sarcasm
checkpoint. Loading it showed a **32×32×3** input — a CIFAR-10 pretraining artifact. The
actual MUStARD++ training loop built a fresh model per fold and never saved weights. No
spectrogram-domain checkpoint exists.

The architecture is fully convolutional + GlobalAveragePooling, so no weight shape depends
on input H×W and the checkpoint *can* be transplanted into a 96×128 model. Whether natural-
photo filters transfer to spectrogram texture was genuinely unknown.

**Decision.** Run both variants — `random_init` and `warm_started` — through the identical
grouped-CV protocol and keep whichever wins on OOF F1. Do not assume; measure.

**Result.** Warm-started won decisively: OOF accuracy `0.841` / F1 `0.875` versus
`0.688` / `0.747`, and converged faster (519s vs 643s). The CIFAR-10 edge and texture
filters transfer usefully across the domain gap, at least at this model size.

**Consequences.**
- Gained: a +15pp result that would have been left on the table if the (reasonable) prior
  that "photos don't transfer to spectrograms" had been trusted.
- Gained: a reusable habit — the A/B costs one extra CV run and removes a guess.
- **Given up**: roughly 10 minutes of CPU per full experiment run, and a permanently more
  complex training script that maintains two code paths.
- **Given up**: an explanation. We know *that* it transfers, not *why* — no ablation was run
  to see which layers carry the benefit.

---

<a id="adr-011"></a>
## ADR-011 — Speaker-grouped cross-validation, always

**Status**: accepted (2026-09-04)

**Context.** RAVDESS has 24 actors speaking two fixed sentences. A random split puts the
same actor in train and validation, and the model learns to recognize *voices*, reporting a
beautiful and meaningless accuracy.

**Decision.** `StratifiedGroupKFold` grouped by `Actor_XX` everywhere, plus a runtime
assertion inside `run_cross_validation` that raises if any group appears on both sides of a
fold — the guard fires even if the `groups` array itself is wrong upstream.

**Consequences.**
- Gained: numbers that mean something. The reported `0.841` is across held-out speakers.
- Gained: a heuristic worth keeping — when a result looks too good on this task, check the
  split before believing it.
- **Given up**: accuracy, straightforwardly. An ungrouped split would report a much higher,
  much more impressive, entirely fake number.
- **Given up**: fold-size balance, since grouping by 24 actors makes folds uneven (208–260
  validation samples).

---

<a id="adr-012"></a>
## ADR-012 — Cross-corpus generalization as the headline metric

**Status**: accepted (2026-09-04)

**Context.** In-corpus OOF answers "did the model memorize 24 actors reading two sentences
in a studio." The product question is whether it survives natural, spontaneous speech.
MUStARD++ was already on disk with a labelled `Arousal` column — a free, genuinely
out-of-domain test set.

**Decision.** Train on RAVDESS, evaluate on MUStARD++ `Arousal` (never trained on), and
report *that* number in the README and the video.

**Result.** Accuracy `0.597` against a `0.504` majority-class baseline on that eval set —
about 9.3pp of real signal surviving the domain shift, versus `0.841` in-corpus.

**Consequences.**
- Gained: an honest generalization estimate, and a finding that directly motivates ADR-003
  (the over-prediction of HIGH is a domain-level baseline shift, which per-speaker
  calibration is designed to absorb).
- Gained: credibility with any evaluator who knows to ask the question.
- **Given up**: the impressive headline. `0.597` is what goes on the slide instead of
  `0.841`, and most viewers will not appreciate why the smaller number is the better one.

---

<a id="adr-013"></a>
## ADR-013 — RAVDESS arousal binarization, with `happy` excluded

**Status**: accepted (2026-09-04)

**Context.** RAVDESS labels eight emotions, not arousal. A binary arousal target has to be
derived. `happy` is high-arousal but positive-valence — including it teaches the model
arousal and valence simultaneously.

**Decision.** HIGH = angry, fearful, disgust, surprised. LOW = neutral, calm, sad.
`happy` (192 clips) excluded from training and retained as a held-out probe. `audioSong`
(1,012 sung clips) excluded entirely as a different prosodic regime. Training set: 1,248
clips, 768 high / 480 low, across all 24 actors.

**Consequences.**
- Gained: a target that isolates arousal rather than entangling it with valence.
- **Given up**: 192 training clips (~13% of available speech data), and any ability to
  distinguish positive from negative high-arousal states — the model cannot tell an excited
  candidate from an anxious one, which matters for a product about interviews.
- **Given up**: balanced classes (61.5% / 38.5%), requiring class weights throughout.

---

<a id="adr-014"></a>
## ADR-014 — MUStARD++ binarization with an ambiguity dead zone

**Status**: accepted (2026-09-04)

**Context.** MUStARD++'s `Arousal` is a 3–9 Likert rating, hard-peaked at 6–7 (297 and 400
of 1,201 rows). A median split lands on the peak and produces a near-degenerate class
balance driven by rater noise.

**Decision.** LOW = `Arousal ≤ 5` (254 clips), HIGH = `Arousal ≥ 8` (250 clips), and drop
the 6–7 band (697 clips) entirely. Documented in the module rather than buried in a
threshold constant.

**Consequences.**
- Gained: a balanced, reliably-labelled 504-clip evaluation set.
- **Given up**: 58% of the available evaluation data, and any measurement of how the model
  behaves on the *middling* cases — which are the majority of real speech, and arguably the
  hardest and most relevant ones. The reported cross-corpus number is therefore an estimate
  on the easy tails, not the full distribution.

---

<a id="adr-015"></a>
## ADR-015 — Abandon institutionally gatekept deception corpora

**Status**: accepted (2026-09-04)

**Context.** DOLOS (1,675 clips) required a request form naming an academic supervisor, and
its terms declare redistribution *or derivation* illegal without express permission —
directly colliding with the hackathon's mandatory public-GitHub-repo requirement.
Bag-of-Lies required a licence signed by a head of institution or registrar. Real-Life Trial
is request-gated and only 121 clips. This gatekeeping is structural to the field: these
corpora are collected under human-subjects protocols, so research-only clauses are the norm,
not a DOLOS quirk.

Separately, filling a supervisor field with fabricated details was never an option.

**Decision.** Move deception corpora off the critical path entirely. Build on RAVDESS
(CC BY-NC-SA, immediate download) and MUStARD++ (already on disk). Send the access requests
anyway, transparently disclosing the hackathon's cash-prize context and asking the licensors
to judge it — treating any approval as upside, never as a dependency.

**Consequences.**
- Gained: an unblocked project with a 26-day deadline, and no licence agreed to under a
  false premise.
- Gained: an honest disclosure trail if any request is later approved.
- **Given up**: deception-labelled data entirely. The system is trained on *arousal*, never
  on deception — which is consistent with ADR-001, but means the project can never make the
  deception claim even if it wanted to.
- **Given up**: acted studio emotion as a proxy for real interview stress, with the domain
  gap that implies.

---

<a id="adr-016"></a>
## ADR-016 — Never commit datasets, derived artifacts, or trained weights

**Status**: accepted (2026-09-04)

**Context.** RAVDESS is CC BY-NC-SA 4.0; MUStARD++ carries its own academic terms. The
hackathon requires a public GitHub repository. Redistribution — not use — is the act that
creates an actual legal claim.

**Decision.** `.gitignore` excludes `databaseAudio/`, all `*.wav`/`*.mp4`, extracted audio
directories, `artifacts/spectrograms/`, `*.keras`/`*.h5`, cached manifests, and `.env`.
Code and methodology are published; data, derived spectrogram images, and trained weights
are not. The README cites RAVDESS via its Zenodo record and states that reproduction
requires the reader's own dataset access.

**Consequences.**
- Gained: a publishable repository with no redistribution exposure, and a clean answer if a
  licensor ever asks.
- **Given up**: reproducibility for anyone who cannot obtain the corpora themselves. A
  reader can inspect every line of code and still cannot rerun the experiment.
- **Given up**: the convenience of shipping a ready-to-run demo checkpoint.

---

<a id="adr-017"></a>
## ADR-017 — Enforce vocabulary discipline in code, not only in prompts

**Status**: accepted (2026-09-04)

**Context.** ADR-001's framing is worthless if it lives only in a system prompt that a later
edit can quietly weaken.

**Decision.** The prohibition is executable.
`tests/unit/test_entities.py::test_to_evidence_dict_never_uses_deception_vocabulary`
asserts that the evidence dictionary — the exact object the Analyst Agent's tools return —
never contains "lying", "lie", "deceptive", "deception", "guilty", or "truthful" in its own
field values. `DECEPTION_VOCABULARY_DISCLAIMER` is a domain constant carried on every
record. `ArousalLabel` is deliberately named for arousal, not stress, at the label level.

**Consequences.**
- Gained: a project value that survives contributor turnover and prompt edits, because CI
  fails when it is violated.
- **Given up**: coverage of the actual output surface. The test guards the *data*; the
  Analyst Agent's natural-language phrasing is still governed by `agents/prompts/analyst.md`
  and remains prompt-level. The strongest guarantee stops at the tool boundary.

---

<a id="adr-018"></a>
## ADR-018 — Quality gates as tests with documented, justified thresholds

**Status**: accepted (2026-09-04)

**Context.** "Minimum acceptable accuracy" is meaningless unless the number is anchored to
something and written down before the run, or it becomes whatever the model happened to
score.

**Decision.** `tests/quality_gates/test_minimum_accuracy_gate.py` reads
`artifacts/metrics/latest.json` and enforces: RAVDESS OOF accuracy and F1 ≥ 0.65 (anchored
to the 61.5% majority-class baseline — the model must beat imbalance exploitation by a real
margin); cross-corpus accuracy must beat *its own* eval set's majority baseline, computed
fresh rather than hardcoded; and cross-corpus accuracy ≥ 0.55 absolute. A fourth gate
asserts the warm-start A/B actually ran and transplanted a non-zero number of layers, so a
config regression cannot silently skip the experiment. Each threshold's reasoning is in the
module docstring, alongside an explicit instruction not to weaken a threshold to make a run
pass.

**Consequences.**
- Gained: an unambiguous pass/fail on the project's own terms, and a regression alarm for
  future model changes.
- **Given up**: the thresholds are still judgment calls made by the person who also built
  the model — defensible, documented, but not externally validated.
- **Given up**: the gates check numbers in a file, not the pipeline that produced them; a
  broken training run that writes plausible metrics would pass.

---

<a id="adr-019"></a>
## ADR-019 — The dashboard degrades gracefully without an API key

**Status**: accepted (2026-09-05)

**Context.** The dashboard's session browsing, timeline, Grad-CAM and spectrogram rendering
need no external service. Only the Analyst Agent chat does. Requiring a key at startup would
have made the entire reviewing surface unavailable — and untestable — without credentials.

**Decision.** `web/backend.py` reads settings lazily, inside the chat endpoint only. With no
key configured, sessions, turns, and artifacts serve normally and `/chat` returns HTTP 503
with an explicit message. Verified with real `curl` against a running `uvicorn`, not only
with `TestClient`.

**Consequences.**
- Gained: the full dashboard was built and verified end-to-end before any key existed, and
  the review surface stays useful for anyone without one.
- Gained: a failure mode that explains itself instead of 500-ing opaquely.
- **Given up**: a startup-time configuration check. A missing or invalid key is discovered
  at first chat request, per request, rather than once at boot.

---

<a id="adr-020"></a>
## ADR-020 — Match transplanted weights positionally by layer type

**Status**: accepted (2026-09-04) · supersedes the original name-keyed implementation

**Context.** The first `load_compatible_weights` matched source and target layers by
`layer.name`. Keras' layer auto-naming is a *process-global* counter: the second
`build_resnet_light()` call in one interpreter yields `conv2d_32`, not `conv2d`. In a 5-fold
loop this silently transplanted **zero** layers from fold 2 onward while the code continued
reporting "warm-started" — which would have invalidated the entire ADR-010 experiment
without any visible error.

**Decision.** Match positionally by layer class (the *n*-th `Conv2D` in the source maps to
the *n*-th `Conv2D` in the target), verifying shape compatibility per layer, and return the
count of layers actually copied. `run_cross_validation` raises if a warm-start was requested
and zero layers were transplanted, and warns on a partial transplant.

**Consequences.**
- Gained: correctness independent of interpreter state, plus a loud failure instead of a
  silent one.
- Gained: a regression test that builds five models before transplanting, reproducing the
  exact k-fold scenario.
- **Given up**: robustness to architectural reordering. Positional matching assumes source
  and target enumerate layers of each type in the same order — true here because both come
  from the same builder, and false the moment someone transplants between genuinely
  different architectures.

---

<a id="adr-021"></a>
## ADR-021 — Discard sub-threshold speech segments before the pipeline sees them

**Status**: accepted (2026-09-05)

**Context.** On the first live run, a **100 ms** segment (`t_start_ms: 13200` →
`t_end_ms: 13300`) — a mic click or room noise, with near-constant pitch
(`f0_std_hz: 2.3`) and an empty transcript — was processed as a real turn and written to the
session as legitimate evidence.

**Decision.** `SyncBus.min_turn_duration_ms` (default `0` for backward compatibility;
`run_interview.py` passes `300`). Segments below the floor return `None` from
`on_speech_stopped()` and do not consume a turn id, so real turns keep a clean sequence.
Discards are counted and logged rather than silently dropped.

**Consequences.**
- Gained: nonsense records stop reaching the evidence store and the dashboard; verified
  working on the next live run, which discarded two blips before the first real turn.
- **Given up**: genuinely short answers. A crisp "yes" or "no" can run under 300 ms and will
  be discarded — in an interview product, a one-word answer to a pointed question is
  arguably the most interesting turn there is.
- **Given up**: the threshold is a fixed guess, not adaptive to the speaker or the noise
  floor.

---

<a id="adr-022"></a>
## ADR-022 — Pair transcripts to turns FIFO, not by "most recent"

**Status**: accepted (2026-09-05) · supersedes the original latest-snapshot pairing

**Context.** Turn `t0007` on a live run had 300 ms of audio (`36300`→`36600`) recorded
against the transcript *"And I'm telling the truth right now, I think."* — an eight-word
sentence that cannot fit in 300 ms. `transcript.user` (final) for a turn can arrive *after*
that turn's `input.speech.stopped`, sometimes not until the next turn's audio has already
finished. Reading "the most recent transcript seen" at `speech.stopped` time attached the
wrong turn's words, consistently one turn late — the exact off-by-one the original code's
`# ASSUMPTION:` comment had flagged as unverified.

**Decision.** `TranscriptPairer`: a FIFO queue where a finalized turn waits for its matching
transcript instead of guessing. Overflow beyond `max_pending=2` flushes the oldest turn with
an empty transcript, so a turn whose transcript never arrives (wordless audio) cannot stall
every pairing behind it. `flush_all()` drains the queue at session end so the last turns of a
conversation are not silently lost.

**Consequences.**
- Gained: correct attribution, and a component testable without audio — including a test
  that reproduces the exact `t0007` ordering.
- **Given up**: immediacy. A turn is no longer finalized at `speech.stopped`; it is finalized
  when its transcript arrives, which adds latency to any real-time display.
- **Given up**: correctness under reordering. FIFO assumes transcripts arrive in turn order.
  That holds for a linear conversation and would break under out-of-order delivery, which we
  have not observed but also cannot rule out.

---

<a id="adr-023"></a>
## ADR-023 — Source protocol defaults from docs and live probes, never intuition

**Status**: accepted (2026-09-05)

**Context.** Four separate protocol details were guessed and four were wrong: the session
config field was `system_prompt`, not `instructions` (rejected outright by the server);
`session.ready` never fires on this account, only `session.updated` with the id nested at
`config.id`; `session.ended` existed in the documented event list but had no dispatch entry,
making a server-initiated close indistinguishable from a crash; and
`min_silence`/`max_silence` defaults of 500/2000 ms were invented when the platform's
documented defaults are 1000/3000 ms — plausibly causing the observed fragmentation of
single sentences into several turns.

**Decision.** `scripts/check_live_connection.py` is a permanent repository tool: it
authenticates, performs the handshake, dumps every raw event with its type, and probes the
LLM Gateway — without opening a microphone or recording anyone. Every protocol constant
carries a comment naming its source (documented default, or observed on a specific date).

**Consequences.**
- Gained: a cheap first check before any live session, and a documented provenance for every
  constant.
- Gained: three real bugs found in a single 15-second run.
- **Given up**: observations are account- and date-specific. `session.ready` not firing is
  true of *this* account today; a paid tier or a config with a greeting may behave
  differently, so both event types must stay handled.

---

<a id="adr-024"></a>
## ADR-024 — Select the LLM by account entitlement probing, not roster reading

**Status**: accepted (2026-09-05) · reconfirmed 2026-09-06 against a live 34-model roster (see [ADR-044](#adr-044)) — same result, a much wider net

**Context.** The LLM Gateway's public model roster lists Claude and GPT families. On this
account's free tier, `claude-sonnet-5`, `claude-haiku-4-5-20251001` and `gpt-4.1` all return
HTTP 400 *"Your account does not have access to this LLM Gateway model"*. The roster
documents what can exist behind the gateway, not what a given account may call.

**Decision.** `check_live_connection.py` probes a candidate list and reports which models
actually respond. `DEFAULT_MODEL` is `qwen3.5-4b-32k-fast` — empirically confirmed working —
with the reasoning recorded at the constant and `AnalystAgent(model=...)` left open for
accounts with broader access.

**Consequences.**
- Gained: a default that works on the account the project actually runs on, and a repeatable
  way to re-derive it if entitlements change.
- **Given up**: model quality for the Analyst Agent. A 4B-parameter model does the grounded
  explanation work that a frontier model would do more fluently — which matters, because
  ADR-004 asks that agent to be careful about hedging and uncertainty, exactly where small
  models are weakest.

---

<a id="adr-025"></a>
## ADR-025 — Flag short-clip arousal scores as unreliable instead of retraining

**Status**: accepted (2026-09-06)

**Context.** A user reported the arousal score as "quite random" — confident short
exclamations ("Yeah, that's for sure.") scored high (0.67–0.97), hesitant hedging ("I
really don't know what to say.") scored low (0.00–0.03), and the *same words* spoken
three times ("Yeah, that's for sure.") scored 0.95, 0.67, and 0.81. Inspecting the saved
session directly (not guessing) showed the actual driver: every turn ≤500ms scored in the
high band regardless of content, while every turn ≥1200ms scored in a lower, more stable
band. `infrastructure/audio/spectrogram.py`'s `MIN_CLIP_SECONDS = 1.0` zero-pads any
shorter clip before spectrogram extraction — a 300ms utterance becomes a spectrogram that
is ~70% silence. RAVDESS training clips are continuous 3–4 second recordings, never
padded. Short live turns are out-of-training-distribution in a way the model was never
calibrated for, and the instability is the visible symptom.

Retraining with padding-aware augmentation, or changing the padding strategy, would
address the root cause but needs new training data and time this pass didn't have.

**Decision.** Make the system honest about it instead. `TurnEvidence.arousal_score_reliable`
is `False` for any turn shorter than `MIN_RELIABLE_AROUSAL_DURATION_MS` (1000ms — set at
the padding boundary itself, not at RAVDESS's native duration, which would flag nearly
every real interview turn). Surfaced as `stress.score_reliable` in the evidence contract;
`agents/prompts/analyst.md` now requires the Analyst Agent to check this *before* saying
anything about the score, and lead with the limitation rather than reporting an unstable
number at face value.

**Consequences.**
- Gained: the dashboard and Analyst Agent stop presenting a coin-flip-unstable number with
  the same confidence as a stable one — directly consistent with [ADR-001](#adr-001)'s
  refusal to overclaim.
- Gained: cheap to ship (one threshold, one property, one prompt update) versus a retrain.
- **Given up**: the root cause is unaddressed. The model still produces the unstable
  number; this only labels it. A one-word "no" to a pointed question — plausibly the most
  interesting turn in an interview — is exactly the kind of turn this flags as unreliable.
- **Given up**: a single global threshold is a blunt instrument. A quieter speaker or a
  noisier room might need a different cutoff than 1000ms; this doesn't adapt.

---

<a id="adr-026"></a>
## ADR-026 — Distinguish "measured zero" from "extraction failed" in prosody

**Status**: accepted (2026-09-06)

**Context.** Inspecting the same session's raw JSON (not just the printed arousal scores)
showed 6 of 10 turns with `f0_mean_hz: 0.0`, `f0_std_hz: 0.0`, `hnr_db` around `-6.5` —
Praat's pitch tracker found zero voiced frames (short and/or quiet utterances) and
`PraatProsodyExtractor` was silently returning a `ProsodyFeatures` full of fallback zeros,
indistinguishable from an (impossible) genuine 0 Hz measurement. Two calibration turns in
that same session had this happen, meaning `BaselineCalibrator` was averaging fabricated
zeros into the baseline mean/std alongside real measurements — corrupting every later
z-score computed against that baseline, silently.

**Decision.** `ProsodyFeatures.f0_detected: bool` (default `True`) records whether Praat
actually found voiced frames. `BaselineCalibrator.add_turn()` skips a turn entirely — not
just its pitch fields — when `f0_detected` is `False`, since jitter/shimmer/HNR come from
the same Praat pitch pass and are equally unreliable when it fails.
`agents/prompts/analyst.md` instructs the Analyst Agent to say pitch data is unavailable
for such a turn rather than describing fabricated numbers as real.

**Consequences.**
- Gained: a corrupted baseline can no longer form silently — a turn that fails pitch
  detection now either doesn't count toward calibration, or (if it drops the count below
  `MIN_CALIBRATION_TURNS`) correctly flips `BaselineProfile.is_reliable` to `False`, which
  the Analyst Agent already knows how to report honestly.
- **Given up**: `speech_rate_syll_s` (librosa onset-based, pitch-independent) is discarded
  along with the pitch fields for a skipped turn — a real signal thrown away for
  implementation simplicity rather than partially salvaged.
- **Given up**: root cause untouched, same as [ADR-025](#adr-025) — this makes failure
  visible and non-corrupting, it does not make Praat detect pitch on a 300ms breathy
  utterance. A more robust pitch tracker, or a minimum-duration gate before attempting
  extraction at all, remains future work.

---

<a id="adr-027"></a>
## ADR-027 — No exception in the live loop may be swallowed without printing

**Status**: accepted (2026-09-06) · supersedes the bare `except (ConnectionError,
KeyboardInterrupt, asyncio.CancelledError): pass` clause

**Context.** `run_interview.py`'s main loop went silent after a single turn **three
separate times** on real runs, each with a different root cause: (1) an unhandled
`websockets.exceptions.ConnectionClosed`, not a builtin `ConnectionError` subclass
([Bug 13](#appendix-a)); (2) a `session.ended` server event with no dispatch entry
([Bug 12](#appendix-a)); (3) — confirmed by the user explicitly ruling out a manual
Ctrl+C — something else entirely, with neither of the first two fixes producing any
output. The prime suspect is a raw `ConnectionResetError`/`OSError` (both builtin
`ConnectionError` subclasses, plausible on an abrupt TCP-level reset rather than a clean
WebSocket close handshake, and a known-annoying category on Windows' asyncio proactor
loop) — silently absorbed by the very `except (ConnectionError, ...): pass` clause that
Bug 13's fix left in place for everything *other* than `ConnectionClosed`.

Each of the first two fixes patched one named exception type. The third occurrence is
the signal that the actual bug class is structural: **any bare `except: pass` (or
`except (Types): pass`) in this loop is a future silent failure**, regardless of which
specific exception eventually lands in it.

**Decision.** Replace the catch-all `pass` with explicit branches that each print
something: `KeyboardInterrupt` → `"[ended] Ctrl+C"`; `asyncio.CancelledError` →
`"[ended] task cancelled"`; `websockets.exceptions.ConnectionClosed` → its code and
reason (unchanged from the Bug 13 fix); and a final `except Exception as e` catch-all
that prints `type(e).__name__` and `str(e)` for anything not named above. The last
branch is the actual fix: it guarantees that a fourth, fifth, or Nth distinct failure
mode still produces a diagnostic instead of requiring another round of "what happened
this time."

**Consequences.**
- Gained: a bug class closed rather than one more instance of it patched. The next
  unexpected disconnect — whatever causes it — prints its type and message instead of
  silence.
- Gained: a reusable review heuristic (recorded in [CLAUDE.md](CLAUDE.md)): a bare
  `except: pass` anywhere in this codebase is a defect on sight, not a style
  preference.
- **Given up**: `except Exception` doesn't catch everything — `BaseException` subclasses
  outside `Exception` (`SystemExit`, and `KeyboardInterrupt`/`CancelledError`, already
  handled above) still need their own branch if a new one is ever relevant. The
  guarantee is "every `Exception` subclass prints something," not "every possible
  interpreter-level signal does."
- **Given up**: still no confirmed root cause for the third occurrence. This makes the
  *next* occurrence diagnosable in one run instead of requiring a fourth patch cycle —
  it does not retroactively explain what already happened.

---

<a id="adr-028"></a>
## ADR-028 — Amortized audio buffer + off-loop scoring, not O(n²) blocking inference

**Status**: accepted (2026-09-06)

**Context.** On a real 14-turn live session, turns 1–9 behaved normally (arousal varying
0.56–1.00 with plausible prosody), then turns 10–14 collapsed to arousal ≈0.00 with
degenerate prosody (`f0_std≈0`, `hnr` near the Praat-failure floor) despite normal
duration (1.1–2.8s, well above [ADR-025](#adr-025)'s reliability floor) — while the
*transcripts* for those same turns stayed correct. Since AssemblyAI transcribes from its
own real-time server-side pipeline on the same underlying stream, correct transcripts
with degenerate local audio meant the drift was specifically in this project's own local
audio path, not the network or the account.

Inspection found two compounding causes:
1. `SyncBus._slice()` called `np.concatenate()` over **every chunk fed since session
   start** on every single turn — O(session-length) work per turn, O(n²) over an
   n-turn session. By turn 10 this was reprocessing several minutes of audio to slice
   out a two-second turn.
2. `_finalize_turn()` called `EvidenceService.build_turn_evidence()` — a CNN forward
   pass, a Grad-CAM forward+backward pass, and Praat pitch extraction, all synchronous
   and CPU-bound — directly on the asyncio event loop, blocking all incoming Voice
   Agent event processing and outgoing audio transmission for its duration.

Together: processing per turn grew slower as the session went on, the local pipeline
fell behind real time, and — while the loop was blocked scoring turn *N* — the mic
kept feeding real audio into the buffer in real time regardless. By the time processing
resumed, the sample range `SyncBus` associated with a given turn id no longer
corresponded to the same audio AssemblyAI's server had already segmented and
transcribed for that turn. Not model instability — a real-time backlog silently
misaligning two independently-progressing clocks.

**Decision.** Two independent fixes, matched to the two causes:
1. `SyncBus` now uses a pre-allocated, amortized-doubling buffer (`numpy`, the same
   growth strategy CPython's `list` uses) — O(1) amortized append, O(k) slice for a
   k-sample request, never O(audio fed so far) for either operation.
2. `_finalize_turn()` now calls `build_turn_evidence` via `asyncio.to_thread`, so the
   CNN/Grad-CAM/Praat work runs off the event loop and audio/event processing continue
   concurrently while a turn is being scored.

**Consequences.**
- Gained: the mechanism that produced 5 consecutive degenerate turns on a real session
  is closed — both the quadratic cost and the loop-blocking that let it accumulate into
  observable drift.
- Gained: two structural, reusable habits — never re-derive a full history when an
  incremental update suffices ([SyncBus](#adr-007) already centralizes segmentation;
  now it also does so efficiently); never call CPU-bound synchronous work directly on
  an event loop that also owns real-time I/O.
- **Given up**: `asyncio.to_thread` uses Python's default thread pool — safe here only
  because turns are processed strictly one at a time (each `_finalize_turn` call is
  awaited to completion before the next starts); it would need explicit serialization
  if this code ever became concurrent (e.g. scoring while the *next* turn's audio is
  still streaming in).
- **Given up**: no direct evidence (a profiler trace, a reproduced timing measurement)
  that this was in fact the exact mechanism, versus a strong, mechanistically-argued
  circumstantial case built from the symptom pattern (correct transcripts, degenerate
  local audio, hard onset at turn 10, both known-inefficient code paths present). The
  fix is real and closes a real inefficiency regardless; whether it fully explains the
  observed drift is confirmed only by the next long live session not reproducing it.

---

<a id="adr-029"></a>
## ADR-029 — Use AssemblyAI's word-level confidence/timestamps as a second hesitation signal

**Status**: accepted (2026-09-06)

**Context.** A documentation review (`voice-agent-api` and `streaming/transcribe-
streaming-audio` pages) confirmed the Voice Agent's `transcript.user` event carries a
`words[]` array — per-word `confidence` (0-1) and `start`/`end` millisecond timestamps —
alongside the plain text this project was already extracting and discarding the rest of.
Two gaps this closes: `ProsodyFeatures.onset_latency_ms` had been `None` on every turn
since the field was designed (ARCHITECTURE.md always described it as "costs nothing to
compute," but nothing ever computed it); and the acoustic pipeline (Praat/CNN) was the
*only* hesitation signal in the system, with no independent cross-check.

Also checked and explicitly **not** adopted in this pass: AssemblyAI's Sentiment
Analysis and Disfluency Detection are both pre-recorded/async "Speech Understanding" API
features, not part of the Voice Agent or streaming path — they would require a separate
post-session batch call, not a live-loop change, and are left as future work.

**Decision.** `application/asr_signals.py` — two pure functions, no I/O:
`mean_word_confidence` (average ASR confidence across a turn's words) and
`onset_latency_ms` (gap between the turn's `t_start_ms` and its first word's `start`).
Wired into `run_interview.py` via a new `TranscriptPayload(text, words)` that
`TranscriptPairer` now carries through its existing FIFO queue instead of a bare `str` —
`TranscriptPairer` was generalized to `TranscriptPairer[T, P]` with a
`default_payload_factory` (defaulting to `lambda: ""`, so every pre-existing caller and
test needed no behavior change, only a type-subscript arity fix). `ProsodyFeatures` is
frozen, so the two new values are patched in via `dataclasses.replace()` after
`build_turn_evidence` returns — the same externally-populated pattern `onset_latency_ms`
was always designed around, just finally exercised.

**Consequences.**
- Gained: `onset_latency_ms` is populated for the first time since it was designed, and
  a genuinely independent hesitation signal (AssemblyAI's recognizer, not our own DSP)
  the Analyst Agent can cross-reference against the acoustic score — they can and will
  disagree, which is informative rather than a bug.
- Gained: `TranscriptPairer`'s generalization is reusable for any future richer payload
  without touching its (already well-tested) FIFO/overflow/flush logic.
- **Given up**: `onset_latency_ms` here is narrower than ARCHITECTURE.md's original
  description (leading silence *within* the turn's own audio, not the full "agent
  finished asking → user began answering" gap, which would need the agent's
  `reply.done` timestamp — not wired up). Labeled explicitly in the docstring and the
  Analyst Agent's prompt rather than left to be assumed.
- **Given up**: relies on the same clock-alignment assumption ADR-025/ADR-028's
  investigations already flagged as unverified — AssemblyAI's word timestamps and
  `SyncBus`'s locally-derived `t_start_ms` are treated as sharing a comparable origin.
  Guarded (negative latency returns `None` rather than a nonsensical number) but not
  proven against live data yet.
- **Given up**: Sentiment Analysis and Disfluency Detection remain unexplored — real,
  documented AssemblyAI features that could add value, deliberately left out of this
  pass's scope rather than half-implemented.

---

<a id="adr-030"></a>
## ADR-030 — Context-injection grounding, because the only reachable model can't call tools

**Status**: accepted (2026-09-06) · supersedes tool-calling as the *default* grounding
mechanism; [ADR-004](#adr-004)'s guarantee is unchanged

**Context.** The dashboard's chat returned HTTP 500 on a real question. The cause,
reproduced directly against the gateway: `qwen3.5-4b-32k-fast` — the *only* model this
trial account can reach — returns HTTP 400 *"model qwen3.5-4b-32k-fast does not support
tools"*. Probing all 12 plausible models with `tools` enabled returned either that error
or *"your account does not have access"* for every single one. The public roster page
states that all models support `tools` and `tool_choice`; empirically, on this account,
none usable does. Same class of finding as [ADR-024](#adr-024) — the roster documents
the platform, not the account, and now also not the capability.

This breaks the Analyst Agent's entire retrieval mechanism, which [ADR-004](#adr-004)
built on tool calling.

**Decision.** Add a second grounding mode and default to it.
`render_session_evidence(session)` pre-renders every turn's evidence — transcript,
score, `score_reliable`, prosody (with the pitch-not-detected warning inline),
baseline deviations — as a compact text block injected into the system message, with a
closing instruction that the model may discuss nothing else. `AnalystAgent` gains
`grounding_mode` (`"tools"` | `"context"`), validated at construction, selected via
`ASSEMBLYAI_LLM_GROUNDING_MODE` (default `"context"`). In context mode no `tools` key is
sent at all, so the gateway accepts the request.

ADR-004's actual guarantee survives intact: the model can still only speak about data
this system put in front of it. What changes is *when* the data is selected — the whole
session up front, instead of on demand. For interview-sized sessions (~3.4KB of rendered
evidence for an 8-turn session) this fits trivially in a 32k window.

**Consequences.**
- Gained: a working Analyst Agent on the account the project actually has, verified
  end-to-end against the live gateway with a real saved session.
- Gained: the tool path is retained and still tested, so an upgraded account gets
  on-demand retrieval by flipping one env var — no code change.
- **Given up**: scalability of the grounding. Context injection is fine for 8-20 turns
  and will not be for a 200-turn transcript; tool calling degrades gracefully there and
  this does not.
- **Given up**: the model can no longer *choose* what to look up, which was a small but
  real part of the XAI story ("it went and fetched turn 7's evidence"). It now receives
  everything and selects rhetorically rather than mechanically — a weaker audit trail of
  what it actually consulted.

---

<a id="adr-031"></a>
## ADR-031 — Enforce the no-deception vocabulary rule in code, on the agent's output

**Status**: accepted (2026-09-06) · closes the gap [ADR-017](#adr-017) explicitly left open

**Context.** ADR-017 put ADR-001's vocabulary rule in the system prompt, guarded the
evidence *data* with a test, and recorded the limitation honestly: *"the test guards the
data; the Analyst Agent's natural-language phrasing is still prompt-level. The strongest
guarantee stops at the tool boundary."*

On 2026-09-06 that stopped being theoretical. Asked why a turn scored high, the live
agent wrote *"the candidate was likely suppressing their initial statement (the
'lie')"* and *"we cannot rely on jitter or shimmer to detect deception here"* — and
separately asserted a pitch-tracking failure the evidence did not contain. Asked the
adversarial version (*"Was he lying?"*), it violated on both its first and second
attempts. A 4B model — the only one available ([ADR-030](#adr-030)) — does not reliably
honour a negative instruction.

For a product whose entire pitch is *"it refuses to claim deception"*, rendering an
answer that speculates about lying is worse than rendering no answer at all.

**Decision.** Move the rule into code. `BANNED_DECEPTION_TERMS` becomes a domain
constant (single source of truth, previously duplicated in a test).
`application/vocabulary_guard.py` provides `contains_violation()` — word-boundary
matching so "believe"/"relief" don't false-positive, with the system's own sanctioned
disclaimer phrasings stripped before checking so the agent isn't punished for stating
the rule it follows. `AnalystAgent.ask()` now: answers → checks → on violation, issues
exactly one corrective turn → re-checks → returns `SAFE_FALLBACK_ANSWER` if it still
violates, and counts interventions in `vocabulary_violations`.

Verified against the live model with a deliberately adversarial question: two
violations, both caught, safe fallback returned, zero banned terms in the output.

**Consequences.**
- Gained: the project's central promise is now a code guarantee on the output surface,
  not an instruction the model may ignore — and it's tested against the verbatim
  sentence that broke it.
- Gained: `vocabulary_violations` makes "the model is fighting its constraints" visible
  to an operator instead of silently absorbed ([ADR-027](#adr-027)'s principle applied
  to model behaviour).
- **Given up**: a blunt instrument. Term matching cannot catch a paraphrase that
  implies deception without using any listed word ("their story doesn't hold up"), so
  this raises the floor rather than closing the surface. It is a guard, not a proof.
- **Given up**: answer quality in edge cases. A legitimate answer that happens to quote
  a candidate saying "I'm not lying" would be refused. Given the asymmetry of harms
  here, refusing a good answer beats rendering a harmful one — but it is a real cost.
- **Given up**: one extra round-trip (and its cost/latency) whenever a violation fires.

---

<a id="adr-032"></a>
## ADR-032 — Give the Analyst Agent the Grad-CAM as facts, not as a file path

**Status**: accepted (2026-09-06)

**Context.** Asked *"what impacted the high arousal in turn 4?"*, the live agent
answered *"I do not have enough information to definitively explain the cause"* and then
speculated vaguely. The user's critique was blunt and correct: **this is not doing XAI.**

The cause was not evasiveness. The XAI chain had a gap at its last metre.
`GradCAMExplainer` computes exactly the thing that explains a score — which region of
the spectrogram the CNN attended to — `EvidenceService` writes it to a PNG, and the
dashboard shows it to a human. But the component whose *entire job* is verbalising that
explanation received only `gradcam_png`: **a file path a text model cannot open**. The
system computed its own explanation and then withheld it from its own explainer.

**Decision.** `application/gradcam_summary.py` converts the heatmap into citable
numbers: peak frequency and half-maximum band (mel bin → Hz), where in the clip the
attention sits, how concentrated it is, and — the one that matters most here —
`attention_on_padding`, the share of attention that landed on zero-padded silence rather
than real audio. `describe_gradcam` renders that as one plain sentence, carried on
`TurnEvidence.gradcam_description`, included in the evidence contract and in
`render_session_evidence`, with `analyst.md` instructed to lead with it.

`attention_on_padding` also turns [ADR-025](#adr-025)'s short-clip artifact from an
inference into a measurement: if the model attended to fabricated silence, that *is* the
explanation of the score, and the agent can now say so.

The heatmap is now computed unconditionally rather than only when saving PNGs.
`SpectrogramExtractorPort` gained `min_clip_seconds` so `EvidenceService` can compute
the real-audio fraction without importing the concrete extractor — the first draft did
import it, inverting [ADR-005](#adr-005)'s dependency arrow.

**Result.** The same question now returns: attention at 376 Hz (~246–730 Hz band, early
in the clip), pitch mean +17.5 SD and pitch variability +14.3 SD above the speaker's own
baseline, speech rate +1.7 SD — followed by the correct caveat that these are
non-specific signals.

**Consequences.**
- Gained: the project's central differentiator actually works. "It explains why" stopped
  being a claim about an image nobody reads to the model.
- Gained: a measurable padding diagnostic, closing the loop on the short-clip problem.
- **Given up**: the summary describes *where the model looked*, which is not the same as
  *why the speaker's voice changed* — a distinction the prompt now states explicitly,
  because the gap between them is exactly where overclaiming would creep back in.
- **Given up**: mel-bin→Hz conversion is reimplemented locally to keep the application
  layer dependency-light, so it can drift from librosa's if the extractor's mel
  parameters change. Nothing currently detects that drift.

---

<a id="adr-033"></a>
## ADR-033 — Make the vocabulary guard assertion-aware, not term-matching

**Status**: accepted (2026-09-06) · supersedes [ADR-031](#adr-031)'s matching strategy;
its *decision to enforce in code at all* stands unchanged

**Context.** ADR-031's guard matched banned terms with a handful of hardcoded
exemptions. Within hours it was refusing legitimate answers: asked a completely innocuous
question, the agent produced a careful answer, the guard fired twice, and the reviewer
got the safe-fallback refusal instead. The guard was flagging the agent for **stating the
very rule it was following** — "I cannot determine whether the speaker was lying",
"this does not indicate deception". ADR-031's own `SAFE_FALLBACK_ANSWER` failed its own
guard.

Blocking a correct explanation defeats the XAI purpose as surely as rendering a harmful
one does — and it did so immediately after [ADR-032](#adr-032) was added specifically to
make explanations better.

**Decision.** The distinction that matters is **where the negation attaches**, checked
per sentence in three ordered passes:
1. Negating the *honesty itself* — "was not truthful", "wasn't being honest" — asserts
   deception in a disclaimer's clothing → violation, negation notwithstanding.
2. Otherwise, negating the *claim* — "cannot determine whether they were lying", "no
   evidence of deception", "not an indicator of lying" → allowed; this is the agent
   doing what it is told.
3. Otherwise, an assertive construction ("was lying", "indicates deception",
   "suppressing the lie") or a bare banned term → violation.

Pinned by 16 parametrised cases split between refusals that must pass and assertions
that must fail, both drawn from real generated text.

**Consequences.**
- Gained: zero guard interventions on the answer that motivated ADR-032, while every
  assertion in the test set is still caught — including "was truthful", since claiming
  someone *was* honest is equally a truthfulness determination.
- Gained: the guard's own fallback text now passes it, which the previous version did not.
- **Given up**: more machinery for a rule that remains, as ADR-031 already conceded, a
  floor and not a proof. A paraphrase that implies dishonesty without any listed word
  ("their story doesn't hold up") still passes.
- **Given up**: sentence splitting is regex-based, so an assertion spanning a clause
  boundary the splitter mishandles could slip through.

---

<a id="adr-034"></a>
## ADR-034 — Leave no gaps: mode-aware prompt, explicit presence/absence, stall guard

**Status**: accepted (2026-09-06)

**Context.** Three separate live failures in one sitting, all the same underlying shape:
**wherever the context left a gap, the model filled it.**

1. Asked to explain a turn, the agent replied *"Let me call `get_turn_evidence` for turn
   t0008."* and stopped — a narrated intention, no answer. `analyst.md` still instructed
   it to call tools, but [ADR-030](#adr-030) had moved the deployment to context mode
   where no tools exist. Telling a model to use a mechanism that isn't there is a prompt
   bug, not a model failure.
2. On a session predating [ADR-032](#adr-032), the agent asserted *"the Grad-CAM
   attention for this turn focused on the low-frequency range"* — invented, because the
   line was simply **omitted** when the field was absent. Backfilling the real data
   showed that turn's attention was at **4768 Hz**: not merely unsupported, wrong.
3. For a turn whose pitch was measured at 113.8 Hz, the agent announced *"pitch was not
   detected for this turn"* — inventing a limitation, because availability was only
   **implied** by the presence of numbers rather than stated.

The model hallucinates in both directions: it invents capabilities it lacks and
limitations it doesn't have. What all three share is that the context said nothing
explicit, and nothing explicit is an invitation.

Two more instances surfaced the same day, extending the pattern from *silence* to
*contradiction* and *unexplained units*:

4. For a turn whose pitch tracking failed, the context said "these z-scores are
   placeholders, do not describe any of them as real" — and then **listed them anyway**.
   The model resolved the contradiction by inventing a concrete value, reporting
   "f0_mean was very low (117.4 Hz)" for a turn whose evidence contained no pitch
   number at all. Disclaiming data while still handing it over is not a safeguard.
5. Given `speech_rate_z=+0.18`, the model reported it as "1.8 times higher than the
   candidate's average". A z-score is not a ratio, and nothing said what it was.

**Decision.** Three changes on one principle — say everything outright:
- `analyst.md` is now mode-neutral about *how* evidence arrives; `AnalystAgent` appends
  an `ACCESS MODE` section describing the actual mechanism. Context mode states plainly
  that there are no tools and that announcing a call produces no answer.
- `render_session_evidence` states absence explicitly ("model attention: NOT AVAILABLE …
  do not describe or guess it") **and** presence explicitly ("pitch detection:
  SUCCEEDED — the values below are real measurements"), rather than letting either be
  inferred from the shape of the text.
- `application/answer_validation.py` detects a stalled announcement (short + references
  retrieval + first-person intent) and nudges once. A stall is a non-answer, not a
  harmful one, so unlike [ADR-033](#adr-033)'s guard it retries rather than refuses.
- `scripts/backfill_gradcam_summaries.py` re-derives summaries for the 46 turns recorded
  before ADR-032, from their already-saved heatmap PNGs — the information was never
  lost, only stored in a form the summariser hadn't seen.
- Pitch-derived z-scores are **withheld** when pitch tracking failed, rather than listed
  under a disclaimer; only `speech_rate_z` (pitch-independent) survives. Withholding
  contradictory data beats labelling it.
- A `HOW TO READ THESE NUMBERS` legend states that z-scores are standard deviations from
  that speaker's own baseline, not ratios or percentages.
- `BANNED_DECEPTION_TERMS` gained the honesty axis (`honest`, `honesty`, `dishonest`,
  `dishonesty`, `truthfulness`, `untruthful`) and the blanket `rather than` refusal
  marker was narrowed to bare abstract nouns — a real answer had characterised the
  speaker's utterance as "an acoustic stress signal rather than a truthful response",
  which the broad marker waved through.

**Result on the failing question.** The same prompt now returns the reliability caveat,
the real attention band (4768 Hz), the real shimmer (47.14%), and a correct
acknowledgement that pitch detection succeeded — zero stalls, zero vocabulary
interventions.

**Consequences.**
- Gained: a reusable rule for grounding a small model — *state presence and absence,
  never imply either* — that generalises past these three instances.
- Gained: 46 historical turns become explainable, so a demo isn't restricted to sessions
  recorded after today.
- **Given up**: verbosity. Every turn now carries availability statements whether or not
  anything is missing, which costs context budget for sessions with many turns —
  directly against [ADR-030](#adr-030)'s scaling limit.
- **Given up**: the stall detector is heuristic (length + retrieval reference + intent
  phrase). A stall phrased differently slips through, and a genuine short answer that
  hits all three signals would be needlessly retried.
- **Given up**: none of this makes the 4B model reliable. It still produced a domain
  error in the final answer ("shimmer measures duration jitter" — it measures amplitude
  variation), which grounding hygiene cannot fix. That needs a better model, which
  [ADR-024](#adr-024) established this account cannot reach.

---

<a id="adr-035"></a>
## ADR-035 — The dashboard reads evidence through the domain layer, not the session file

**Status**: accepted (2026-09-06)

**Context.** `GET /api/sessions/{id}` returned the persisted JSON verbatim while the
Analyst Agent built its context block from `TurnEvidence` entities loaded through
`to_evidence_dict()`. Two readers, two schemas, one underlying file. Any field the domain
layer normalises on load — `f0_detected` inferred for legacy turns ([ADR-026](#adr-026)),
`score_reliable` derived from `duration_ms` ([ADR-025](#adr-025)), backfilled Grad-CAM
summaries — existed for the agent and not for the UI.

That divergence is the exact shape of the failure this project is built to avoid: a
reviewer reads a number on screen, asks the agent about it, and gets an answer computed
from a different number. Neither is wrong; they disagree, and nothing in the system can
say which the reviewer should trust.

**Decision.** The endpoint loads the session as entities and serialises turns through
`to_evidence_dict()`, keeping session-level keys from the raw file:

```python
raw = load_session_raw(session_id)
session = load_session_entity(session_id)
return {**{k: v for k, v in raw.items() if k != "turns"},
        "turns": [turn.to_evidence_dict() for turn in session.turns]}
```

`to_evidence_dict()` is now the single evidence schema. The frontend reads the nested
shape (`stress.score`, `prosody_raw.f0_detected`, `xai.gradcam_description`) — the same
fields, under the same names, that reach the agent's context block.

**Consequences.**
- Gained: what the reviewer sees and what the agent reasons over cannot drift, because
  they are the same serialisation. A derived field added to the domain layer reaches both.
- Gained: `test_to_evidence_dict_never_uses_deception_vocabulary` ([ADR-001](#adr-001))
  now transitively guards the dashboard payload too, not just the agent's input.
- **Given up**: the endpoint pays a full entity load per request where it previously did
  a file read. Irrelevant at session sizes of tens of turns; it would not be at thousands.
- **Given up**: the frontend is coupled to the domain serialisation. Renaming a field in
  `to_evidence_dict()` breaks the UI silently — there is no schema test spanning the two.

---

<a id="adr-036"></a>
## ADR-036 — The Analyst Agent is a persistent panel, not a tab

**Status**: accepted (2026-09-06)

**Context.** The first dashboard put the Analyst chat behind a tab, competing with the
evidence views for the same region. Testing it, the reviewer's report was that it was
"not even possible to talk to the agent" — the chat was reachable, but reaching it meant
hiding the turn you wanted to ask about. Reading evidence and interrogating it are one
task, and the layout had split them into two modes.

The interrogation *is* the product ([ADR-002](#adr-002), [ADR-030](#adr-030)); the
numbers alone are the part the project explicitly refuses to let stand on its own.

**Decision.** Three fixed columns: sessions left, timeline and turns centre, turn detail
*and* the Analyst chat both permanently in the right rail. The reviewer can read a score
and ask about it without a mode switch. Tabs remain only where the alternatives are
genuinely alternatives (SPEECH TURNS / VOICE ANALYTICS).

Separately, static assets carry an explicit version query (`style.css?v=4`). The
`no_store_static` middleware from Appendix A row 32 only takes effect once a client
fetches under the new header — it cannot evict an entry a browser already holds, which
is precisely the case after any earlier visit. A changed URL can.

**Consequences.**
- Gained: the chat is visible in every screenshot of the tool, which is what the tool is.
- Gained: cache-busting is independent of server restarts, closing the half of row 32's
  staleness trap that `no-store` alone leaves open.
- **Given up**: vertical space. The chat log is capped (`max-height: 46vh`), so long
  exchanges scroll within a short panel rather than expanding.
- **Given up**: the layout assumes a wide viewport. Three fixed columns have no mobile
  story; a narrow screen is not usable, and no breakpoint was written for one.
- **Given up**: `?v=` is bumped by hand. Forgetting to bump it on a frontend edit
  reintroduces exactly the bug it was added to prevent — a build step would not have
  this failure mode.

---

<a id="adr-037"></a>
## ADR-037 — Precompute session-scope facts; never ask the model to rank or count

**Status**: accepted (2026-09-05) · extends [ADR-034](#adr-034) from turn scope to
session scope

**Context.** [ADR-036](#adr-036) added three starter prompts to the chat panel, putting
generic session-wide questions one click from any reviewer. Run against a real 14-turn
session, two of the three produced false statements:

- *"What drove the highest score?"* → **"the highest arousal score across the session was
  0.958 in turn t0007"**. The true maximum was **0.997 at t0004**. The model had ranked
  fourteen rows of prose by eye and got it wrong.
- *"Which turn stands out most?"* → named t0004 correctly, then cited a
  **"contrast between t0039 and t0040"**. Neither exists; the ids run t0001–t0014.

Every per-turn fact was already stated correctly and explicitly, exactly as ADR-034
requires. The gap was one level up: nothing in the context stated what the maximum *was*,
or which ids existed. ADR-034's lesson — a gap gets filled with invention — applies to
derived facts as much as to absent ones. Ranking and counting across many rows is the
canonical thing a language model should not be asked to do from prose.

**Decision.** `_render_session_summary` computes session-scope facts in Python and states
them at the top of the context block:

- the exhaustive list of turn ids, with *"the ONLY ids that exist"* and an instruction to
  refuse any id outside it;
- the highest arousal **among reliable scored turns**, labelled as the answer to "which
  stands out" / "what scored highest";
- the highest among *all* scored turns, printed only when it differs — with its
  `score_reliable` flag and an instruction to keep the caveat attached, so a 400 ms blip
  cannot be laundered into the headline;
- the mean across reliable scored turns;
- the **membership**, not merely the count, of the unreliable set and the pitch-failed
  set. Counts alone were not enough: given them, the model still grouped by eye and
  called t0010 (2600 ms, reliable) low-confidence.

**Consequences.**
- Gained: verified live against the session that produced both failures — the maximum is
  now reported as t0004 at 0.997, and across re-runs no answer cited a turn id that does
  not exist.
- Gained: the reliability caveat is structural. The model cannot present the loudest
  number in a session as the finding without the flag that undermines it.
- **Given up**: the aggregates encode an editorial judgement — that the *reliable*
  maximum is the headline. A reviewer who wants the raw maximum foremost is reading
  against the grain of the prompt.
- **Given up**: context length grows with turn count, and the id list grows linearly. Fine
  for interview-sized sessions; a 200-turn session would need a different shape.
- **Given up**: this fixes ranking, not all cross-turn reasoning. One run still described
  t0009 (0.866) as "the next highest turn" when unreliable t0007 (0.958) sits between —
  a loose characterisation rather than a false value, and not currently guarded.

---

<a id="adr-038"></a>
## ADR-038 — Seed the chat panel with starter prompts, and treat them as shipped surface

**Status**: accepted (2026-09-05) · extends [ADR-036](#adr-036)

**Context.** Making the Analyst panel prominent ([ADR-036](#adr-036)) left it prominent
and empty. A reviewer meeting the dashboard for the first time can see that the box
matters without knowing what it will usefully answer, and the panel's own size then works
against it — a large blank area reads as an unfinished feature.

**Decision.** Three clickable starter prompts sit under the log until the first message,
then hide: *"Which turn stands out most?"*, *"What drove the highest score?"*, *"How
reliable is this session?"*. They fill the input and submit through the form's own
handler, so there is one send path and not two.

They are chosen to demonstrate the grounding guarantee rather than to flatter it: each
asks for a session-wide judgement the agent must derive from evidence, and the third
invites the agent to report the session's *weaknesses*.

**Consequences.**
- Gained: the panel teaches its own purpose, and a demo has a defined opening move
  instead of an improvised one.
- Gained, unexpectedly and most valuably: putting three generic questions one click away
  made them easy to actually run — which is how the session-scope invention in
  [ADR-037](#adr-037) was found. A feature that is hard to invoke is a feature whose
  failures stay hidden.
- **Given up / live risk**: the prompts invite rapid successive clicking, and the free
  tier rate-limits at roughly this cadence — running all three in sequence returned
  HTTP 429 on the third. The error path is actionable (Appendix A row 23), but a judged
  demo that clicks three chips in a row will show an error on the third. **Mitigation for
  a live demo: pace the questions, or expect and narrate the limit.** Not fixed in code;
  a client-side cooldown would hide a real constraint rather than remove it.
- **Given up**: the prompts steer what gets asked. A reviewer who only ever clicks chips
  exercises three paths out of many, and the untested ones stay untested.

---

<a id="adr-039"></a>
## ADR-039 — A guard is not accepted until it has been made to fail

**Status**: accepted (2026-09-05)

**Context.** Eight frontend contract tests were written for [ADR-036](#adr-036) and all
eight passed on the first run. Passing was the only evidence they were real, and for one
of them it was wrong evidence: the schema guard searched `app.js` for `turn.score`, while
the failure it existed to catch produces `t.score`. Breaking the invariant deliberately
proved it — seven mutations failed loudly, the eighth passed. A test that cannot fail is
worse than no test, because it is counted as coverage.

A second instance in the same file: a `\b` written inside a non-raw Python string became a
literal backspace character (`\x08`), silently weakening the regex it belonged to. Both
defects were invisible to `pytest`, and both were caught the same way.

This is the same disease as Appendix A rows 1, 2, 5 and 6 — trusting a name, a label, or
a green tick over the artifact — arriving in the place meant to protect against it.

**Decision.** Every new guard is mutation-tested before it counts: break the invariant it
claims to protect, confirm *that specific test* fails, restore, confirm the suite is
green, and confirm the restored file is byte-identical to the original. A guard that
survives its own mutation is rewritten, not kept.

The restore check is not ceremony. Mutations are applied with `sed -i` against real
source files, and a botched restore ships a mutation.

**Consequences.**
- Gained: the frontend guards are known-live, not assumed-live. Eleven mutations run
  across two rounds; every one produced the expected failure after the weak test was
  rewritten.
- Gained: a concrete acceptance criterion for "is this test worth keeping", replacing
  the judgement call that let the weak one through.
- **Given up**: real time per guard. Roughly a minute of mutate/run/restore for each
  invariant, which is why this is a rule for guards protecting a catalogued bug — not
  for every assertion in the suite.
- **Given up**: it verifies that a test *can* fail, not that it fails for the right
  reason. A guard could pass its mutation for an unrelated reason; only reading the
  failure message rules that out, and that step stays manual.
- **Given up**: mutating files in place is destructive by construction. The backup-and-
  diff step is mandatory, and on Windows a `/tmp` backup is readable by Git Bash but not
  by the Windows Python interpreter — a mismatch that once made a clean restore *look*
  like a corrupted one.

<a id="adr-040"></a>
## ADR-040 — Wire the Interview Agent's reply back to a human, for the first time

**Status**: accepted (2026-09-05)

**Context.** Building a browser capture surface meant looking closely at
`voice_agent_client.py` for the first time since ADR-030, and turned up something no
prior live-run investigation had considered: `reply.started` / `reply.audio` /
`reply.done` — the event family SKILLS.md documented from day one (2026-09-04) as
carrying the Interview Agent's synthesized speech — were never wired into the dispatch
table. `transcript.agent` (the agent's reply as text) *was* wired, but no composition
root ever passed it a handler. The result: every live session that has ever run sent the
candidate's mic audio to the Voice Agent and surfaced *nothing* of its reply — not
spoken, not written — to whoever was on the microphone.

This reframes, as a plausible (not confirmed) explanation, the still-open mystery in
ARCHITECTURE.md Section 7b: sessions that ended after a single turn with no diagnostic.
A candidate who says something and then hears and sees nothing back has every reason to
sit in silence — which looks, from the transport layer, identical to a dropped
connection. Nobody had considered this because every investigation so far treated it as
a protocol/transport bug, never as "the human had no idea the agent replied."

**Decision.** Three new callback fields (`on_reply_started`, `on_reply_audio`,
`on_reply_done`) and three new dispatch entries in `voice_agent_client.py` --
zero-risk, additive, and covered by the same FakeTransport pattern as every other event
(`test_reply_started_dispatches_to_handler` and two siblings). `LiveInterviewRunner`
wires all three: `on_agent_transcript` now actually receives a handler (status log +
`on_agent_reply_text` observer callback), and `_on_reply_audio` decodes the payload and
forwards it through `on_agent_reply_audio` — both consumed by the browser capture path
in ADR-041, giving a human ears and eyes on the conversation for the first time.

**Consequences.**
- Gained: `scripts/run_interview.py`'s operator now sees `[agent] <text>` printed for
  the first time, at zero cost — no dependency, no audio hardware needed for the text
  half. It still cannot play the agent's voice; see ADR-041 for where that lands.
- Gained: a candid, testable hypothesis for a previously "still open" bug, instead of
  another silent gap.
- **Given up**: the exact shape of `reply.audio`'s payload is UNCONFIRMED against a
  real connection — `_on_reply_audio` assumes a base64 `"audio"` key by symmetry with
  `input.audio`'s own shape, marked `# ASSUMPTION:` in code. This is exactly the kind
  of guess that was wrong twice before (`instructions` vs `system_prompt`,
  `session.ready` vs `session.updated`) — the first real session either confirms it or
  produces a `[reply.audio] event carried no 'audio' field: {...}` diagnostic instead of
  a silent failure, by design (`test_on_reply_audio_logs_instead_of_crashing_when_
  field_missing`).
- **Given up**: this ADR only wires the plumbing. Nothing plays the audio until
  ADR-041's browser page; the terminal script still has no speaker output.

---

<a id="adr-041"></a>
## ADR-041 — Browser-based capture, decoupled from sounddevice, sharing the runner

**Status**: accepted (2026-09-05)

**Context.** `scripts/run_interview.py` required a local Python environment, mic
drivers, and a terminal — real barriers for a hackathon judge who wants to see the
product work, and for anyone testing it without cloning the repo's exact toolchain. The
request was a web interface for the capture side (Frente 1) to match the dashboard
already built for the review side (Frente 2, ADR-035/036).

Two paths were available: duplicate `LiveInterviewRunner`'s orchestration inside
`web/backend.py`, or extract it so both composition roots share one implementation.
Duplication was rejected outright — that file carries ~15 documented, hard-won bug
fixes (ADR.md Appendix A rows 8-31), each discovered on a real live run. A second copy
means a second chance to reintroduce every one of them independently, silently, with no
test connecting the two.

**Decision.** `LiveInterviewRunner` moved from the script into
`application/live_interview_runner.py` and became audio-source-agnostic via a small
`AudioSource` Protocol (`start(feed)` / `stop()`). Two adapters implement it:
`infrastructure/audio/mic_source.py`'s `MicAudioSource` (real `sounddevice.InputStream`,
the terminal path) and `web/live_capture.py`'s `WebSocketAudioSource` (no thread of its
own — the FastAPI receive loop already reads the socket and calls `push()` directly).
`scripts/run_interview.py` shrank to nine lines of actual composition.

The browser side: `web/static/capture.html` + `capture.js` + `capture-worklet.js`.
`getUserMedia` leads into an `AudioWorkletNode` (`capture-worklet.js`, running on the
audio thread) that resamples to exactly `SAMPLE_RATE_HZ` via linear interpolation
whenever the browser doesn't honour the requested `AudioContext` rate (SKILLS.md Hard
Rule 5), producing int16 PCM binary frames sent over one WebSocket to `/ws/interview`.
The same socket carries JSON status/turn/agent-text events and raw PCM16 binary frames
for the agent's reply audio (ADR-040) back to the browser, scheduled on a running
playback cursor for gapless output. `web/live_capture.py` owns the framing and, since
`LiveInterviewRunner.run()` gained an injectable `connect` parameter for this ADR, its
full lifecycle — handshake validation, event relay, graceful shutdown, session save --
is exercised by `tests/unit/test_live_capture.py` against a scripted fake AssemblyAI
transport, with no live key and no real browser.

**Consequences.**
- Gained: a candidate can run a full interview from any modern browser tab; the
  terminal path is now one adapter among two, not the only implementation.
- Gained: 21 new tests at the runner/dispatch layer (all passing, mutation-verified per
  ADR-039) plus 9 at the WebSocket-relay layer and 13 static contract guards on the new
  page — `scripts/run_interview.py`'s orchestration logic has real unit coverage for
  the first time in this project's history, as a side effect of being forced to make it
  audio-source-agnostic.
- **Given up — the honest limit of what was tested**: the full real loop — actual
  microphone, actual browser `AudioWorkletNode`, actual AssemblyAI connection, actual
  speaker output — has never run. This is the same boundary ARCHITECTURE.md already
  names for the terminal script's mic path, now inherited by a second, more complex
  surface (an AudioWorklet's resampling math is not something `node --check` or a
  FakeTransport can confirm sounds correct). `capture.js` and `capture-worklet.js`
  state this in their own comments; a real session, run by a human, is the remaining
  step before this ships in a demo.
- **Given up**: the candidate's own words appear in the browser's conversation log only
  once their turn finalizes (via the "turn" event's `.transcript`), not live word-by-
  word — no separate live-transcript hook was wired for the candidate's own speech,
  only for the agent's.
- **Given up**: two independent `SESSIONS_DIR` constants (`backend.py`'s and the
  runner's, each derived from its own file's location) land on the same real directory
  in production but must be patched separately in a test — caught only because an
  early version of `test_live_capture.py`'s fixture patched one and left six real files
  in the actual repo (Appendix A row 38).
- **Given up**: `artifacts/sessions/*.json` and `artifacts/turn_artifacts/**/*.png`
  were never in `.gitignore` before this pass — real and RAVDESS-derived candidate data
  was one `git init` away from being committable (Appendix A row 39, fixed alongside).

---

<a id="adr-042"></a>
## ADR-042 — Never report success without checking for it

**Status**: accepted (2026-09-05)

**Context.** The first real end-to-end check of ADR-041's browser capture path —
connect for real, against a live ASSEMBLYAI_API_KEY, immediately end the session —
reported `{"type": "session_saved", ...}` back over the socket. No file existed at the
path it named. `LiveInterviewRunner.run()`'s `finally` block (flush pending turns, call
`_save_session()`) sat only around the `while True: handle_next_event()` loop — one
indent shallower than the method's actual lifetime. A cancellation landing during
`configure()` or before `pump_task` was created (exactly what happens when a client
sends `end_session` within milliseconds of `session_started`, as a real test did)
propagated straight out of `run()` without ever reaching that `finally`. Meanwhile
`web/live_capture.py` awaited the cancelled task, saw no exception it didn't already
handle, and unconditionally sent `session_saved` — true whenever the race didn't fire,
false exactly when it did, with nothing distinguishing the two cases in the message
itself.

This is the sharpest instance yet of the pattern named in ADR-001/ADR-034's "never claim
what isn't verified": not a wrong number or an invented detail, but a plain
success/failure flag reporting the wrong one. It surfaced in the first minute of live
testing against a real service, which is exactly why ARCHITECTURE.md insists a claim
isn't settled until it has run for real — every test in `test_live_capture.py` used a
scripted fake transport, and every one of them passed against the buggy version too,
because none of them cancelled the session before its event loop started.

**Decision.** Two independent fixes, addressing cause and symptom separately:
1. `run()`'s `try/finally` now wraps the whole body — from `configure()` through the
   event loop — not just the loop. `pump_task` is declared `None` beforehand so the
   `finally` can check it without a `NameError` if cancellation lands before it's ever
   created. Cancellation at any point now guarantees flush + save before the coroutine
   can exit, matching Python's own guarantee that a `finally` runs during any exception
   propagation, cancellation included.
2. `web/live_capture.py` no longer *assumes* success from a clean task exit — it checks
   whether the file the runner claims to have written actually exists, and reports
   `{"type": "error", ...}` naming the missing path if not. This is deliberate
   redundancy: fix (1) closes the specific race that was found, but a literal disk
   write failure (permissions, full disk) is a different failure mode fix (1) does not
   touch, and check (2) catches that too, for the same reason — the message should
   never assert more than what was verified.

`test_run_saves_the_session_even_when_cancelled_before_the_event_loop_starts` reproduces
the exact race with a fake transport whose `send()` never returns (letting a test
suspend and cancel `run()` mid-`configure()` on demand, no timing luck required);
confirmed to fail against the pre-fix code and pass after. A second test forces
`_save_session()` itself to fail (`SESSIONS_DIR` pointed at a plain file) and confirms
`web/live_capture.py` reports `error`, not `session_saved`, when nothing was written.

**Consequences.**
- Gained: the one code path this whole project treats as its integrity boundary — "a
  claim in this system is backed by something real" — now holds for the capture
  surface's own success signal, not only for the analysis it produces.
- Gained: both regression tests reproduce their failure deterministically (a stalled
  `send()`, a file where a directory should be) rather than relying on real timing, so
  neither is a flaky test disguised as a fast one.
- **Given up**: this was found by running the real system once, briefly, against a real
  key — not by any of the 270 tests that existed before that moment. It is a direct,
  concrete instance of the concept-implementation-validation loop doing the one thing
  a mocked test suite structurally cannot: it does not know what it did not think to
  fake.
- **Given up**: the fix's own scope is bounded by what was found. Fix (2)'s file-
  existence check is specific to `_save_session()`'s known output path; a different
  silent-failure shape elsewhere in the same shutdown sequence is not automatically
  covered by this pattern, only by applying the same "verify, don't assume" instinct
  the next time something is added there.

---

<a id="adr-043"></a>
## ADR-043 — A wrong guess must degrade, never flood

**Status**: accepted (2026-09-05)

**Context.** The first real multi-turn conversation through ADR-041's browser capture
page froze the tab after about three exchanges. The visible session log (screenshotted
by the user) showed the cause directly: a wall of `A` characters — a base64-encoded
audio chunk — followed by `'type': 'reply.audio', 'timestamp': ...`. ADR-040's
`_on_reply_audio` guessed the payload lived under an `"audio"` key; every real event
proved that guess wrong, so every single reply.audio chunk hit the fallback branch,
which — per its own comment, "log the raw event... temporarily" — interpolated the
*entire* event, base64 blob included, into a status message. `capture.js`'s
`logStatus()` appended each as an uncapped DOM text node. A single TTS reply is dozens
of chunks; a few exchanges were enough to make the tab unresponsive.

This is a second instance of the *general* pattern ADR-042 named — a claim or an action
proceeding on an unverified assumption — but the failure mode here is different in kind:
not a false positive report, but an unbounded diagnostic feeding on its own uncertainty.
The more wrong the guess, the more it fired, the larger each log line, compounding
rather than merely repeating.

**Decision.** Three independent layers, none of which depends on knowing the real key:
1. `_on_reply_audio` now tries several plausible payload keys (`audio`, `delta`,
   `data`, `chunk`, `pcm`, `audio_delta`) instead of one guess — a real chance of
   working immediately rather than shipping a second single guess with the same
   failure shape.
2. When none match, the diagnostic reports field **names and shapes only**
   (`{"mystery_field": "str(len=50000)"}`) — never a field's actual value. A wrong
   guess can no longer produce an unboundedly large message no matter what the real
   payload looks like.
3. `LiveInterviewRunner._status()` — the single choke point every diagnostic in this
   class passes through — now truncates to 500 characters regardless of source. This
   is the general fix: it protects every current and future status message, not only
   `reply.audio`'s. `capture.js`'s `logStatus()` adds the same caps independently
   (600 chars/line, 300 lines retained, oldest evicted) — a frontend has no business
   trusting a server to always behave, symptom and cause each get their own guard.

**Consequences.**
- Gained: whichever key `reply.audio` actually uses, this now very likely decodes it
  correctly without needing to learn the exact answer first — and if none of the six
  guesses are right, the next diagnostic will name the real key directly instead of
  drowning it in its own value.
- Gained: no single mis-shaped event, from this endpoint or a future one, can freeze a
  browser tab by volume alone — the two independent caps (backend message length,
  frontend line count) mean a repeating bug degrades to "the log looks truncated,"
  never "the tab stops responding."
- **Given up**: the real key is still, as of this writing, unconfirmed — this trades
  guessing once for guessing several times in parallel, which is more likely to work
  but is not the same as knowing. The first successful playback (or the next diagnostic
  naming the actual key) is what actually closes ADR-040's original ASSUMPTION.
- **Given up**: an event whose real audio field is named something (or shaped
  something) fully absent from the widened candidate tuple — including one that isn't
  a bare string, e.g. nested one level down — still falls through to the same
  bounded diagnostic. Broader still means slower to notice a genuinely new shape.

---

<a id="adr-044"></a>
## ADR-044 — Give the Interview Agent exactly one tool, on the entitlement that actually grants it

**Status**: accepted (2026-09-06)

**Context.** ADR-024 established that this account's LLM Gateway cannot reach a tool-
calling-capable chat-completions model — confirmed again 2026-09-06 with a much wider
net than the original check: `GET /v1/models` returned a live roster of 34 models (not
the 4 originally guessed), and every one of them, tested individually against a real
tool-calling request, returned either "your account does not have access" or (for the
one reachable model, `qwen3.5-4b-32k-fast`) "model does not support tools". The Analyst
Agent's `grounding_mode="context"` default is not a workaround for insufficient
probing; it is now a thoroughly confirmed dead end on this account, specifically for
that product.

The Voice Agent API is a **separate product with separate entitlement**, and had never
been tested for tool-calling at all — every prior session configured `tools=[]`
deliberately, for the correct reason (ARCHITECTURE.md §2's score-blindness guarantee),
which had nothing to do with whether the capability existed. A live probe — one
harmless diagnostic tool, a system prompt instructing the agent to call it immediately
— returned a real `tool.call` event with the exact expected name and arguments on the
first attempt. This account's Voice Agent API supports tool-calling; its LLM Gateway,
on every model available to it, does not.

**Decision.** The Interview Agent is now configured with exactly one tool,
`flag_technical_issue` — the candidate reporting an audio/connection problem (echo,
can't hear, cutting out). `agents/prompts/interviewer.md` instructs calling it only for
that, and not mentioning the tool to the candidate. `_on_tool_call` dispatches on tool
name explicitly and rejects anything else rather than acting on it blindly; the
resulting flag is stored on `InterviewSession.technical_flags` (persisted through
`_save_session`/`from_session_dict`) and reported through the existing `_status`
channel — no new UI surface was built for it. `send_tool_result` closes the loop the
Voice Agent API expects.

Confirmed end to end against the real production `LiveInterviewRunner` (not just the
isolated diagnostic probe): a live session was driven via
`request_reply(instructions=...)` to trigger the tool, and the saved session file
contained `"technical_flags": ["live wiring test"]` — the exact string requested,
having passed through the real API, the real dispatch, and the real save path.

**Consequences.**
- Gained: the Interview Agent's tool-calling capability — a headline feature of the
  hackathon's own Voice Agent API description — is now exercised for real, not left
  configured-but-unverified the way the LLM Gateway's tool path still is.
- Gained: a concrete, demoable claim that is also a structural guarantee: "the agent
  can call tools; it has exactly one, and that one cannot touch anything analytical" —
  provable by reading `FLAG_TECHNICAL_ISSUE_SCHEMA` and by
  `test_flag_technical_issue_schema_names_nothing_analytical`'s banned-term scan.
- Gained: real, if minor, product value — a candidate-reported technical problem is no
  longer lost information; it's attached to the session a reviewer opens later.
- **Given up**: `_voice_session` is a new mutable reference `LiveInterviewRunner` holds
  on itself, set and cleared across `run()`'s lifecycle — the first piece of state in
  this class that exists purely so a handler can write *back* to the live session,
  where every previous handler only observed. It is a narrow, single-purpose channel
  (tool-call acknowledgement only) but it is a new category of capability in this
  class, worth watching if a future tool is ever added.
- **Given up**: one tool, one narrow purpose. This does not generalize to "the
  Interview Agent has a tool-calling framework" — adding a second tool means
  deliberately extending `_on_tool_call`'s explicit name dispatch, not registering a
  handler generically; that friction is intentional, per ADR-002.

---

<a id="adr-045"></a>
## ADR-045 — Re-read the docs before building on a working capability; don't move the Analyst Agent to voice

**Status**: accepted (2026-09-06)

**Context.** ADR-044 confirmed the Voice Agent API supports tool-calling on this
account. The natural next question — since the LLM Gateway's tool-calling is
confirmed dead on every reachable model (ADR-024, ADR-044) — was whether the Analyst
Agent itself should move onto a second Voice Agent API connection, gaining real
tool-calling for its evidence lookups instead of the context-grounding fallback, and
letting a reviewer literally talk to it instead of typing into the dashboard's chat.

Before writing any code, AssemblyAI's own documentation was re-read specifically for
the mechanic that would make or break this: whether the agent's generated reply text
can be inspected or gated before its audio starts playing. The vocabulary guard
(ADR-031/033) exists entirely because a small model, unsupervised, has produced real
policy-violating text — the whole point of running it is a synchronous
generate-then-check-then-render step, with nothing reaching the reviewer until the
guard passes.

Two facts settled it, quoted from AssemblyAI's own documentation:

1. **"The server streams `reply.audio` chunks as the LLM generates the response — you
   don't wait for the full reply to start playing."** (Voice Agent API 5-minute
   walkthrough.) Audio synthesis is incremental and concurrent with generation, not a
   separate step after a complete, inspectable text exists.
2. **`execution_mode: "hold"`** — the one documented mechanism for making an agent wait
   — is scoped to tool-call latency ("phone transfers, escalations, long-running
   ops"), not content review: "agent stays silent while the tool runs," then, once
   `tool.result` arrives, resumes generating and speaking immediately. Nothing in the
   documented protocol describes a hook between "text generated" and "audio playing"
   that a caller can use to approve or reject content first.

There is no documented interception point. A vocabulary violation on this path could
be audible before any code of ours ever sees the text that produced it.

**Decision.** The Analyst Agent stays where it is: an LLM Gateway chat client, in
`grounding_mode="context"`, unchanged by this investigation. The tool-calling capacity
ADR-044 confirmed on the Voice Agent API is used only for the Interview Agent's one
tool, where the guard this ADR is protecting doesn't apply — `interviewer.md` already
forbids the Interview Agent from ever discussing analysis at all, so there's no
generated content there that ADR-031/033's guard needs to catch.

Also fixed as a side effect of this investigation: `reply.audio`'s payload key is
CONFIRMED `"data"` (not the originally guessed `"audio"`, ADR-043) — quoted directly
from the walkthrough's own example handler (`event['data']`). `_REPLY_AUDIO_
CANDIDATE_KEYS` now lists `"data"` first as a known fact; the remaining candidates
stay as a safety net for anything this specific search didn't surface.

**Consequences.**
- Gained: a concrete, documented reason not to build something that would have taken
  real engineering time and shipped a real regression (ADR-001's core, non-negotiable
  guarantee, broken by architecture rather than by a bug). Finding this by reading
  docs cost an hour; finding it after building the feature would have cost the feature.
- Gained: `reply.audio`'s key is no longer an open question — a genuine unknown became
  a confirmed fact, and the code and its comments now say so plainly instead of
  hedging.
- **Given up**: the more original, more impressive-sounding demo idea (reviewer talks
  to the Analyst Agent by voice) is not happening for this submission. The safer,
  already-built text-chat Analyst Agent remains the only way to interrogate evidence.
- **Given up**: this is a documentation-based conclusion, not a live-tested one — no
  code was written to actually observe `transcript.agent` and `reply.audio` arrival
  order on a real connection with a guard-triggering response. If AssemblyAI's
  documentation is incomplete or wrong here (their own docs elsewhere have contained
  gaps this project found the hard way — ADR-024's four-model roster, for one), this
  conclusion could be revisited with a live test in `check_live_connection.py`'s
  style. It was not, given the size of the risk being avoided and the time left before
  the deadline.

---

<a id="adr-046"></a>
## ADR-046 — Make TensorFlow an opt-in cost, not a fixed one, for the dashboard deployment

**Status**: accepted (2026-09-06)

**Context.** Preparing to host `web/backend.py` publicly (for the submission's required
"Application URL") surfaced that `backend.py` imported `live_capture` at module top —
which imports `LiveInterviewRunner`, which imports `KerasArousalClassifier`, which
imports TensorFlow. Every process running this file paid TensorFlow's full import cost
and memory footprint, whether or not that process would ever actually run live capture.
Measured directly: importing `backend` with no `ASSEMBLYAI_API_KEY` set took the same
~15-20s and 500MB+ RAM as a live-capture-capable process, for a dashboard that never
touches the classifier at all — session data is pre-computed JSON, artifacts are
pre-rendered PNGs, and the Analyst Agent is a plain `httpx` client to the LLM Gateway.

This matters concretely for hosting: most free tiers cap RAM well under what a resident
TensorFlow import needs, and a public deployment intended only to showcase the
dashboard (ADR-045's decision not to expose live capture with a real key to arbitrary
internet traffic makes this the intended shape of that deployment) would risk an
out-of-memory crash for a dependency it structurally cannot use.

**Decision.** `from live_capture import run_capture_session` moved from module scope in
`backend.py` to inside `interview_capture()`'s body, after the `MissingConfigError`
check. A deployment with no API key configured returns from that check before reaching
the import line, so it never loads `live_capture`, `LiveInterviewRunner`, or TensorFlow.
Confirmed by direct measurement, not assumed: import time dropped to 0.57s with
`tensorflow` absent from `sys.modules` entirely.
`test_live_capture_import_is_lazy_not_module_level` guards the import site itself
(inspects `backend.py`'s own source for a module-level import line), not just the
measured outcome, so a future edit re-adding it at module scope fails loudly.

**Consequences.**
- Gained: a dashboard-only public deployment can run on a free/low-memory tier that a
  TensorFlow-resident process could not.
- Gained: faster cold starts for every deployment shape, including local dev when only
  browsing past sessions.
- **Given up**: the first real live-capture request on a running dashboard-only process
  now pays TensorFlow's import cost at that moment instead of at startup — a real
  latency spike on the first call, invisible until then. Acceptable here because that
  path is deliberately not exposed on a public host in the first place (ADR-045).
- **Given up**: `live_capture`'s own transitive TensorFlow dependency is unchanged;
  this defers the cost, it doesn't remove it. A deployment that legitimately needs live
  capture (a properly provisioned host with its own key) still needs the RAM.

---

<a id="adr-047"></a>
## ADR-047 — A public dashboard deploy ships with no API key by default

**Status**: accepted (2026-09-06)

**Context.** The hackathon submission requires a public "Application URL," not just a
GitHub repository. Two decisions already made this session shape what that URL should
serve: ADR-045 (live capture is not exposed with a real key to arbitrary internet
traffic - no documented way to gate its output before it's spoken) and the same
reasoning extends to the Analyst Agent's chat endpoint, whose free-tier rate ceiling is
explicitly documented as low (Appendix A row 23) and would be trivially exhausted by a
few strangers, or a few judges, hitting it in the same window.

Before proposing a hosting platform, a full build-and-run of the intended deployment
was simulated locally in an isolated venv (a clean `git archive` checkout, not this
machine's own environment) rather than assumed to work from reading the config alone.
That simulation surfaced two real, independent problems, neither hypothetical:

1. **A pip bug, not a compromised package.** `pip install -r requirements.txt` failed
   with "THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE... someone
   may have tampered with them" - alarming wording, investigated rather than dismissed
   OR panicked over. Every named package downloaded completely, at a plausible file
   size, with no errors; the mismatch traced to an unnamed dependency's PyPI metadata
   blob, and the "Got" hash changed on every retry (three different values across three
   runs) while "Expected" stayed fixed - a pattern consistent with a resolver-side
   verification bug in pip 25.0.1, not a consistent tampered artifact (which would
   reproduce identically). Upgrading pip to 26.2.1 made the failure disappear entirely
   on an unmodified requirements.txt. `render.yaml`'s build command now upgrades pip
   before installing anything, so Render's own base image - which could ship an
   equally old pip - doesn't hit the same wall on the one deploy attempt this project
   gets before the deadline.
2. **A real dry-run gap in the plan itself.** The first simulation attempt tested only
   half the build command (`pip install -r requirements.txt`) and skipped
   `pip install -e .`, then failed at startup with `ModuleNotFoundError: No module
   named 'voicestress'` - a real omission in the verification, not a red herring; it
   would have failed identically on Render's actual infrastructure had it not been
   caught here first.

**Decision.** `render.yaml` deploys `web/backend.py` via Render's Python runtime with
no `ASSEMBLYAI_API_KEY` declared. Confirmed end to end in the isolated simulation:
dashboard routes and the demo session return 200, `/chat` returns a clean 503 with an
actionable message, matching the same tested behavior `MissingConfigError`'s handler
already provides locally. A judge reviewing the hosted URL sees the full dashboard,
the demo session, and its Grad-CAM/prosody breakdown; live capture and live chat are
demonstrated in the video instead, run locally with a real key.

**Consequences.**
- Gained: a public URL that cannot run up API costs for arbitrary visitors, because it
  has nothing capable of doing so - not a rate limit bolted onto a live capability, but
  the capability's absence.
- Gained: the pip-upgrade step and the two-part build command are both now verified,
  not assumed, against a real isolated build - the wording of the pip error ("someone
  may have tampered with them") was exactly alarming enough that skipping the
  investigation in favor of assuming it was a local fluke would have been the wrong
  call.
- **Given up**: a judge cannot try live capture or ask the Analyst Agent a question
  directly on the hosted URL - only in the video, or by running the project locally
  with their own key. This is the direct cost of ADR-045's decision, paid here rather
  than reopened.
- **Given up**: this was verified in one isolated local venv, not on Render's actual
  infrastructure - a genuinely different environment (OS, base image, network path)
  could still surface something this simulation didn't. The first real deploy is still
  the first real test of Render specifically, consistent with this project's own
  standard for what "confirmed" means elsewhere (ARCHITECTURE.md Section 7b).

---

<a id="adr-048"></a>
## ADR-048 — Confirmed live: the deployed no-key default fails safely from a real browser

**Status**: accepted (2026-09-06)

**Context.** ADR-047's decision — ship the public Render URL with no
`ASSEMBLYAI_API_KEY`, so live capture rejects cleanly instead of exposing a cost-bearing
surface — was verified locally in an isolated venv before deploying, but not yet
against the actual hosted instance from a real browser. The first real attempt produced
`[ws] connection error` / `code=1006` with no further detail: consistent with a raw,
unexplained failure, and worth investigating rather than assuming either "it's broken"
or "it's fine."

Two independent checks resolved it. First, connecting directly to the deployed
`/ws/interview` with a plain Python WebSocket client (bypassing the browser entirely)
succeeded immediately and returned the exact intended message
(`"Voice Agent not configured: ..."`) — proving the deployed app and Render's
WebSocket routing both work correctly. Second, comparing the user's two browser
attempts line by line was the actual diagnostic: the first attempt's log jumped
straight from `[mic] AudioContext running...` to `[ws] connection error`, with no
`[ws] connected, handshake sent` line ever appearing — meaning the connection never
completed opening at all. The second attempt, run shortly after (by which point the
Python test had already reached the instance), printed `[ws] connected, handshake
sent` followed by the correct `[error] Voice Agent not configured...` message, then
closed. Same code, same deployment, two different outcomes purely as a function of
whether the free-tier instance was already warm.

**Decision.** No code change. This confirms, rather than alters, ADR-047: the
intended safe-default behavior works correctly end to end, including from a real
browser against the real hosted URL — once the instance is warm. The remaining,
accepted gap is operational, not architectural: a Render free-tier instance that has
spun down after inactivity can fail a WebSocket upgrade attempt on the very first hit
with a raw, unexplained `1006`, because (unlike a stateless HTTP request, which Render
can hold and retry transparently while the container boots) a half-open WebSocket
upgrade during that boot window appears to be dropped rather than queued.

**Consequences.**
- Gained: the no-key safety design (ADR-045/047) is now confirmed working in the one
  environment that actually matters — the real public URL, hit by a real browser —
  not just asserted from a local simulation.
- Gained: a wrong-looking symptom (a bare connection error, no message) was correctly
  diagnosed as an infrastructure timing gap rather than either dismissed or mistaken
  for a code defect — by comparing exact log sequences rather than guessing from the
  error code alone.
- **Given up**: this remains a real rough edge for anyone's first visit to the public
  URL right after a period of inactivity — not just for `/ws/interview` (which is
  deliberately non-functional there anyway) but potentially for the dashboard's own
  first request too. No fix is proposed here: paying for a plan that doesn't spin down
  is the actual fix, and isn't warranted for a demo URL whose primary content (the
  dashboard, the committed session) tolerates a slow first load far better than a
  live conversation would.

---

---

<a id="appendix-a"></a>
## Appendix A — Bug catalogue

Every bug that reached running code, and what now prevents its recurrence. Each row's
guard is a real test in the suite (287 passing at time of writing).

| # | How it showed up | Root cause | What prevents recurrence |
|---|------------------|------------|--------------------------|
| 1 | Warm-start reported success while training from scratch, folds 2–5 | Keras layer auto-naming is a process-global counter; name-keyed weight matching silently matched nothing | Positional-by-type matching ([ADR-020](#adr-020)); `test_transplant_still_works_after_several_models_built_in_same_process` builds 5 models first; `run_cross_validation` raises on a zero-layer transplant |
| 2 | Evidence keys `jitter_local_z` / `speech_rate_syll_s_z` instead of the documented `jitter_z` / `speech_rate_z` | Deriving dict keys by stripping unit suffixes off field names with chained `.replace()` | Explicit field→key table in `value_objects.py`; `test_deviation_sign_and_magnitude` asserts the contract key |
| 3 | `GradCAMExplainer` raised `AttributeError` on construction | `@dataclass(slots=True)` rejects attributes assigned in `__post_init__` that aren't declared fields | Plain `@dataclass` where derived state is attached; `slots=True` reserved for immutable value objects; covered by `test_gradcam_*` |
| 4 | A correct guard looked like a failing test | Test fixture varied only `f0_mean` across calibration turns, leaving five features at zero variance — `BaselineProfile.is_reliable` was right to reject it | Realistic multi-feature fixture; the guard itself is pinned by `test_identical_turns_yield_degenerate_std_and_are_not_reliable` |
| 5 | Architecture doc described a warm-start from a spectrogram-domain checkpoint that does not exist | `resnet1_prelayer1.keras` is a 32×32×3 CIFAR-10 pretraining artifact; the real training loop never called `.save()`. Variable names were trusted over the artifact | Load and inspect artifacts directly; `test_param_count_matches_known_inherited_checkpoint` pins the real 6,578-parameter shape |
| 6 | Parameter count documented as 19,384 | `.summary()`'s total includes Adam optimizer-state slots; `count_params()` reports 6,578 weights | Same test as above, asserting `count_params()` explicitly |
| 7 | Live server rejected session config: `invalid_format` | Config field guessed as `instructions`; the real field is `system_prompt` | [ADR-023](#adr-023); `check_live_connection.py`; `test_configure_sends_tools_and_turn_detection` pins the field name |
| 8 | Session id never captured on a live connection | `session.ready` does not fire on this account; only `session.updated`, with the id at `config.id`, not top-level | Both event types dispatched; `test_session_updated_captures_session_id_from_nested_config` |
| 9 | LLM Gateway returned HTTP 400 for a documented model | Public model roster ≠ per-account entitlement | [ADR-024](#adr-024); candidate-model probing in `check_live_connection.py` |
| 10 | A 100 ms mic click written to the session as a real scored turn | No minimum-duration floor between VAD events and the acoustic pipeline | [ADR-021](#adr-021); `test_short_blip_below_minimum_duration_is_discarded` and three companions |
| 11 | 300 ms of audio recorded against an eight-word transcript | `transcript.user` can arrive after that turn's `speech.stopped`; "most recent transcript" pairing was off by one turn | [ADR-022](#adr-022); `test_transcript_arriving_after_next_turn_still_pairs_with_correct_turn` reproduces the exact ordering |
| 12 | Session ended after one turn with no explanation printed | `session.ended` had no entry in the dispatch table and was silently dropped | Dispatch entry added; `test_session_ended_event_reaches_registered_handler` |
| 13 | Server-initiated disconnects produced no diagnostic at all | `websockets.exceptions.ConnectionClosed` is **not** a subclass of built-in `ConnectionError` (confirmed via its MRO), so the `except` clause never caught it | Explicit `except ConnectionClosed` branch printing `e.code` / `e.reason` |
| 14 | Single sentences fragmented across four turns ("I" / "I think." / "gonna lie now.") | `min_silence_ms` defaulted to a guessed 500 ms; the platform's documented default is 1000 ms | Defaults corrected to the documented values, with provenance recorded at the parameter |
| 15 | `.env` written but never read | `config.py` read `os.environ` directly with no dotenv loading | `load_dotenv` at import with `override=False`, so real environment variables still win |
| 16 | Arousal scores swung 0.67–0.97 on short turns regardless of spoken content, looking "random" | `MIN_CLIP_SECONDS=1.0` zero-pads short clips before spectrogram extraction; the CNN never saw padded, mostly-silent inputs during training on unpadded 3-4s RAVDESS clips | [ADR-025](#adr-025); `TurnEvidence.arousal_score_reliable`; `test_duration_and_reliability_flag_short_clip` |
| 17 | Baseline z-scores computed from a mix of real and fabricated data, silently | `PraatProsodyExtractor` returned `f0_mean=0.0` etc. when Praat found no voiced frames, indistinguishable from a real measurement; `BaselineCalibrator` averaged it in | [ADR-026](#adr-026); `ProsodyFeatures.f0_detected`; `test_turn_with_undetected_pitch_is_skipped_not_corrupting_baseline` |
| 18 | A live session ended after one turn a THIRD time, with zero diagnostic output despite Bugs 12 and 13's fixes; user confirmed no manual Ctrl+C | Prime suspect: a raw `ConnectionResetError`/`OSError` (builtin `ConnectionError` subclasses) from an abrupt TCP reset, absorbed by the `except (ConnectionError, ...): pass` clause Bug 13's fix left in place for everything besides `ConnectionClosed` | [ADR-027](#adr-027): `pass` replaced with explicit, always-printing branches per exception type plus a catch-all `except Exception` — root cause still unconfirmed, but no future occurrence can go silent |
| 19 | On a 14-turn live session, turns 10-14 scored arousal≈0.00 with degenerate prosody (`f0_std≈0`) despite normal duration, while their transcripts stayed correct | `SyncBus._slice()` re-concatenated the entire session's audio history on every turn (O(n²) over the session); combined with synchronous CNN/Grad-CAM/Praat work blocking the event loop per turn, later turns' processing fell behind real time and the local sample bookkeeping drifted out of sync with what the server had already segmented and transcribed | [ADR-028](#adr-028): amortized-doubling buffer (`SyncBus`) + `asyncio.to_thread` offloading of `build_turn_evidence`; `test_many_turns_stay_correct_at_scale`, `test_buffer_capacity_grows_amortized_not_per_chunk` |
| 20 | Dashboard chat returned HTTP 500 in the browser with no usable diagnostic | The gateway rejected the request (`"model qwen3.5-4b-32k-fast does not support tools"`) and the un-caught `httpx.HTTPStatusError` surfaced as a bare FastAPI 500 | [ADR-030](#adr-030) (context grounding, so the request is accepted at all) + the endpoint now returns 502 carrying the gateway's own message; `test_chat_returns_502_with_gateway_detail_not_bare_500` |
| 21 | The live Analyst Agent speculated about the speaker "suppressing their lie" and about detecting "deception" — violating the project's central rule | The rule existed only in the system prompt; a 4B model (the only one this account can reach) does not reliably honour a negative instruction | [ADR-031](#adr-031): `vocabulary_guard` checks the generated answer, retries once, then refuses; `test_detects_the_real_violating_sentence_from_the_live_run` pins the verbatim failure |
| 22 | Sessions recorded before `f0_detected` existed presented Praat's fabricated 0.0 fallbacks to the dashboard and agent as real measurements | `ProsodyFeatures.from_dict` defaulted the missing key to `True`; 5 of 9 turns in one real session were affected | `from_dict` now infers `False` when `f0_mean_hz` and `f0_std_hz` are both 0.0 (a 0 Hz mean pitch is physically impossible); an explicit flag still always wins. `test_legacy_session_without_f0_detected_infers_false_from_zero_pitch` |
| 23 | Chat showed a raw `{"message":"too many requests for this action"}` gateway dump mid-conversation | Free-tier rate limit (HTTP 429) surfaced unhandled; the ADR-031 guard's retry doubles request count on a violation, making it more likely | Backend returns a 429 explaining the limit and what to do; frontend renders it as guidance rather than an error dump. `test_chat_returns_actionable_429_not_raw_gateway_json` |
| 24 | Asked what drove a turn's score, the agent answered "I do not have enough information to explain the cause" — failing its core XAI job | The only Grad-CAM output reaching it was `gradcam_png`, a file path a text model cannot open; the explanation was computed, saved, and withheld from the explainer | [ADR-032](#adr-032): heatmap summarized into citable facts (peak band in Hz, position, `attention_on_padding`) and injected into the agent's context; `test_locates_peak_frequency_band` and 9 companions |
| 25 | The ADR-031 vocabulary guard refused a legitimate answer to an innocuous question, twice in a row | Term matching flagged the agent for *stating the rule it follows* ("I cannot determine whether they were lying"); even the guard's own fallback text failed its own guard | [ADR-033](#adr-033): three-pass, sentence-level, negation-scope-aware check; 16 parametrised cases split between refusals that must pass and assertions that must fail |
| 26 | Asked to explain a turn, the agent replied "Let me call `get_turn_evidence` for turn t0008." and stopped — no answer at all | `analyst.md` still instructed tool calls after ADR-030 moved the deployment to context mode, where no tools exist | [ADR-034](#adr-034): `ACCESS MODE` section injected per mode + stall detector that nudges once; `test_detects_the_real_stalled_answer_verbatim` |
| 27 | The agent asserted a Grad-CAM finding ("focused on the low-frequency range") for a session that had no Grad-CAM data; the real value was 4768 Hz | Absent fields were **omitted** from the rendered context rather than marked absent — silence invited invention | [ADR-034](#adr-034): explicit "NOT AVAILABLE … do not describe or guess it"; plus `scripts/backfill_gradcam_summaries.py` for the 46 affected turns; `test_rendered_evidence_states_gradcam_absence_explicitly` |
| 28 | The agent announced "pitch was not detected" for a turn whose pitch was measured at 113.8 Hz — inventing a limitation | Availability was only *implied* by the presence of numbers; nothing stated it | [ADR-034](#adr-034): "pitch detection: SUCCEEDED — the values below are real measurements" stated positively; `test_rendered_evidence_states_pitch_success_positively` |
| 29 | The agent reported a concrete "f0_mean was very low (117.4 Hz)" for a turn whose evidence contained no pitch value at all | The context disclaimed the pitch-derived z-scores as placeholders and then listed them anyway; the model resolved the contradiction by inventing a matching raw value | [ADR-034](#adr-034): pitch-derived z-scores withheld entirely when tracking failed, keeping only `speech_rate_z`; `test_pitch_derived_z_scores_are_withheld_when_pitch_failed` |
| 30 | "an acoustic stress signal rather than a truthful response" passed the vocabulary guard | The blanket `rather than` refusal marker (meant for "arousal rather than truthfulness") also waved through phrasings that characterise the speaker's answer as untruthful; `honest`/`honesty` weren't banned terms at all | [ADR-034](#adr-034): marker narrowed to bare abstract nouns, honesty axis added to `BANNED_DECEPTION_TERMS`; both phrasings pinned in `test_assertions_about_honesty_are_violations` |
| 31 | `speech_rate_z=+0.18` was reported to the reviewer as "1.8 times higher than the candidate's average" | Nothing in the context explained that a z-score is a standard-deviation offset rather than a ratio | [ADR-034](#adr-034): `HOW TO READ THESE NUMBERS` legend; `test_rendered_evidence_explains_how_to_read_z_scores` |
| 32 | Two fixes (Markdown rendering, vocabulary guard) appeared not to work after being verified correct on disk and in tests | Two independent staleness traps: the browser replayed a cached `app.js`, and uvicorn without `--reload` holds imported Python modules in memory, so backend edits never took effect | `no_store_static` middleware sends `Cache-Control: no-store` for JS/CSS/HTML; `test_static_assets_are_sent_with_no_store`. The Python side is inherent to uvicorn — documented in SKILLS.md §6b as "restart the server after backend edits" |
| 33 | The redesigned dashboard came back visually wrong and functionally broken — the reviewer reported it was "not even possible to talk to the agent" | Two independent frontend failures no Python test could see: the Analyst chat was placed behind a tab, one click from invisible, and nothing checked that the markup, the script and the stylesheet still agreed on ids and class names | [ADR-036](#adr-036): chat promoted to a permanent right-rail panel; `tests/unit/test_frontend_contract.py` cross-checks every `getElementById` against the markup, every class against the stylesheet, every evidence leaf against its [ADR-035](#adr-035) group, and every asset against one shared `?v=` — each guard mutation-tested by breaking the invariant and confirming the failure |
| 34 | Two of the dashboard's own starter prompts produced false statements: "the highest arousal score across the session was 0.958 in turn t0007" (the real maximum was 0.997 at t0004) and a cited "contrast between t0039 and t0040" in a session whose ids run t0001-t0014 | Per-turn facts were stated explicitly per [ADR-034](#adr-034), but session-scope facts were not: nothing said what the maximum was or which ids existed, so the model ranked fourteen rows of prose by eye and invented ids to fill an unenumerated space | [ADR-037](#adr-037): `_render_session_summary` precomputes the id list, the reliable maximum, the mean, and the membership of the unreliable and pitch-failed sets; `test_summary_names_the_true_maximum`, `test_summary_enumerates_the_only_valid_turn_ids`, `test_summary_names_the_unreliable_and_pitchless_turns_not_just_their_count` |
| 35 | A newly written frontend guard passed on every run and could not fail: it searched `app.js` for `turn.score` while the drift it existed to catch produces `t.score`. Counted as coverage for the ADR-035 schema contract, it protected nothing | The test was accepted because it was green. Nothing had ever demonstrated it could go red, and the string it matched was written from memory of the code rather than from the failure mode | [ADR-039](#adr-039): every guard is mutation-tested before it counts — break the invariant, confirm that specific test fails, restore, verify byte-identical. The guard was rewritten to check each evidence leaf carries its group (`test_evidence_leaves_are_read_through_their_group`), then re-mutated |
| 36 | A regex in a new test silently lost its word-boundary anchor: `\b` written inside a non-raw Python string was written to disk as a literal backspace byte (`\b`) | The test file was generated by a script that built its content with a normal `'''...'''` string, where `\b` is an escape sequence, not two characters. `pytest` was green either way — a weaker regex still matched the correct code | Grepping the written file for control characters is now part of the same pass as the mutation test; both are in [ADR-039](#adr-039). Generated test content uses raw strings |
| 37 | A test fixture named `_real_session_shape` asserted the live session's counts ("10 of 14 met the duration floor") against turns whose durations were the helper's defaults, not the recording's — it failed on the first run, claiming 7 where reality had 6 | The fixture claimed to mirror a real session but only copied the scores, leaving durations and pitch outcomes at their defaults. Cf. row 4, where a wrong fixture made a correct guard look broken; here it made a correct guard look wrong in the opposite direction | The fixture was corrected to the recording's real durations and `f0_detected` values — the expectations were left alone, since they were the true numbers. A fixture that claims to mirror real data now asserts that data's own figures, so drift fails loudly |
| 38 | An early version of `test_live_capture.py`'s `sessions_dir` fixture patched `backend_module.SESSIONS_DIR` but not the runner's own copy, then left six real session files behind in the actual `artifacts/sessions/` directory across two test runs | `backend.py` and `live_interview_runner.py` each derive their own `SESSIONS_DIR` from their own file's location — two separate module-level constants that happen to resolve to the same real path in production, so patching one silently leaves the other pointed at the real repo | The fixture now patches both explicitly, with a comment naming why; the six leaked files were deleted; `test_end_session_control_message_saves_and_confirms` asserts against the patched path, not the real one |
| 39 | While fixing row 38, `.gitignore` was found to never have covered `artifacts/sessions/*.json` or `artifacts/turn_artifacts/**/*.png` — real and RAVDESS-derived candidate data was one `git init` away from being committable | The existing "derived artifacts" block covered spectrograms, models, and manifests but was never extended to per-session outputs when `scripts/build_demo_session.py` and `run_interview.py` started writing them | Both globs added to `.gitignore` with a comment explaining what they hold and why |
| 40 | A websocket test helper (`_drain_until_bytes`) hung indefinitely instead of failing when a mutation broke the expected message — and the FIRST fix attempt (wrapping `ws.receive()` in a `with ThreadPoolExecutor() as pool:` with `future.result(timeout=...)`) hung too | Starlette's `WebSocketTestSession.receive()` has no built-in timeout. The first fix's `pool.__exit__` calls `shutdown(wait=True)`, which blocks on the same abandoned worker thread still stuck inside the real `.receive()` call — timing out the *wait* did not un-stick the *thread* | Replaced with a daemon `threading.Thread` reporting through a `queue.Queue`: an abandoned worker is simply leaked, and daemon threads don't block process exit. Confirmed by re-running the same mutation that caused the original hang and observing a fast, correct failure instead |
| 41 | The live capture endpoint reported `{"type": "session_saved"}` over the socket for a session whose file was never written to disk — found on the first real end-to-end connection with a live API key | `run()`'s `finally` (flush + save) sat only around its event loop; a cancellation landing during `configure()` or before `pump_task` existed (a client sending `end_session` within milliseconds of `session_started`) propagated out of `run()` without ever reaching it, while the caller unconditionally reported success anyway | [ADR-042](#adr-042): the `try/finally` now wraps `run()`'s entire body; the caller additionally verifies the claimed file exists before reporting success. `test_run_saves_the_session_even_when_cancelled_before_the_event_loop_starts`, `test_reports_error_not_session_saved_when_the_file_was_never_written` |
| 42 | The Analyst Agent's tool-calling code path had never fired against a live model, and the assumption it never would (per ADR-024's original 4-model check) went untested at wider scope for over a day | Re-probing was cheap and never done until asked to "deepen" the integration — a case of accepting a negative result as permanent instead of re-checking when circumstances (time, credits) might have changed | [ADR-044](#adr-044): reconfirmed against a live 34-model roster (still negative for LLM Gateway), which led directly to testing the Voice Agent API's SEPARATE tool-calling entitlement instead — confirmed working on the first live attempt |
| 43 | The live capture page froze the browser tab after ~3 real exchanges; the session log showed why — a wall of base64 characters followed by `'type': 'reply.audio'` | ADR-040's guessed payload key ("audio") was wrong; every real event hit the fallback, which interpolated the ENTIRE raw event (base64 blob included) into a status message, appended as an uncapped DOM node per chunk — dozens of chunks per reply compounded fast | [ADR-043](#adr-043): `_on_reply_audio` tries six candidate keys instead of one guess; the no-match fallback reports field names/shapes only, never values; `_status()` truncates to 500 chars at its one choke point; `capture.js`'s log independently caps line length and count |
| 44 | `_on_reply_audio`'s payload key was left as a best-effort guess-among-several (ADR-043) rather than a confirmed fact, for the time it took to actually read AssemblyAI's own documentation | The first live session that produced audible playback was treated as "good enough" — it worked, so the exact key was never pinned down, even though the real answer was sitting in already-public docs | [ADR-045](#adr-045): confirmed `"data"` from the Voice Agent API walkthrough's own example handler; `_REPLY_AUDIO_CANDIDATE_KEYS` reordered accordingly, `test_on_reply_audio_decodes_base64_under_the_confirmed_real_key` added |
| 45 | Seven files (three composition-root scripts, four integration tests) had a local Windows username and folder layout hardcoded into dataset-path constants, discovered only when auditing the repo for credential/cost risk minutes after the first public GitHub push | The strings were never a credential and so never tripped the pre-commit secret search (which specifically grepped for the API key and generic token patterns) — they are a DIFFERENT class of exposure (identifying, not authenticating), and a search built for one class missed the other | Each constant now derives its path from `Path(__file__).resolve()` relative to the repo, with `VOICESTRESS_HACKATHON_ROOT` as an explicit override — zero personal strings in source, identical behavior on the machine that already had the data. Verified by re-running the full integration suite, which exercises real RAVDESS audio through the new path resolution rather than skipping |
| 46 | Preparing to host the dashboard publicly surfaced that every process running `backend.py` — including a pure dashboard deployment that never runs live capture — paid TensorFlow's full ~15-20s import time and 500MB+ memory footprint | `live_capture` was imported at module top rather than where it's actually used, so the cost was fixed rather than conditional on the feature actually being reachable (no API key, per ADR-045, means it never is on a public host) | [ADR-046](#adr-046): import moved inside the route handler, after the missing-key check; confirmed by direct measurement (0.57s, no tensorflow in sys.modules) rather than assumed; `test_live_capture_import_is_lazy_not_module_level` guards the import site itself |
| 47 | `pip install -r requirements.txt` failed in an isolated dry-run venv with "THESE PACKAGES DO NOT MATCH THE HASHES... someone may have tampered with them" — alarming wording, on a completely unmodified requirements file | pip 25.0.1's hash-verification logic hit a false positive on an unnamed dependency's PyPI metadata; confirmed not a real tamper by three retries producing three different "Got" hashes against the same fixed "Expected" one (a real compromised artifact would reproduce identically) | [ADR-047](#adr-047): upgrading pip to 26.2.1 resolved it entirely; `render.yaml`'s build command now upgrades pip before installing anything, so the one real deploy attempt doesn't hit the same wall on an older base image |

### The pattern across all forty-seven

Bugs 1, 2, 3, 5, 6 came from **trusting names over artifacts** — a layer name, a field
name, a variable name, a summary line. Bugs 7, 8, 9, 12, 13, 14 came from **trusting
documentation prose over observed behaviour**. Bugs 10, 11 came from **assumptions about
real-time event ordering that only a live conversation could test** — and both were
explicitly flagged as `# ASSUMPTION:` in the code before they fired, which is why they took
minutes rather than hours to locate.

The transferable practice: when a fact can be checked by loading the artifact, connecting to
the server, or printing the raw event, check it — and when it cannot yet be checked, write
down that it is unverified, at the line that depends on it.

---

<a id="appendix-b"></a>
## Appendix B — Cross-cutting trade-offs

Five costs recur across the records above and are worth stating as a whole, because they
shape what this project is and is not.

**Honesty over impressiveness.** ADR-001, ADR-012, ADR-014, and ADR-018 each chose the less
flattering option: no deception claim, `0.597` as the headline instead of `0.841`, a
dead-zone eval set that measures only the easy tails, and thresholds fixed before the run.
The project is calibrated to survive a sceptical technical reviewer, not to win a first
impression. That is a deliberate bet on the audience.

**Testability over velocity.** ADR-005 and ADR-006 bought a suite that runs in ~8 seconds
without hardware, credentials, or network — which is what made it possible to build the
entire agent layer before an API key existed. The cost was real: more files, more
indirection, and a swappability argument that remains theoretical with one adapter per port.

**Proven over modern.** ADR-009 reused a 2017-era spectrogram CNN because it worked and was
already validated, knowingly forgoing the accuracy a wav2vec2-class model would bring. In a
26-day window this was correct; as a product decision past the hackathon it is the first
thing to revisit.

**Structural guarantees over policy.** ADR-002 and ADR-017 encode project values as code
that fails CI rather than as documentation that erodes. Both stop at a real boundary: the
Interview Agent cannot receive a score, but the Analyst Agent's *phrasing* is still
prompt-governed. Knowing exactly where the hard guarantee ends is more useful than
pretending it covers everything.

**Guards that also discard signal.** ADR-013 dropped 192 clips, ADR-014 dropped 697,
ADR-021 discards genuinely short answers, and ADR-003 spends the first three turns of every
interview on calibration. Each is defensible individually; together they mean the system
sees less data, and less of the interesting data, than a less careful version would. The
loudest example: a one-word answer to a pointed question is both the most interesting turn
in an interview and the one most likely to fall under the 300 ms floor.

**Every gap closed makes the context more prescriptive.** ADR-032 added the Grad-CAM as
facts, ADR-034 added explicit presence/absence statements and a units legend, ADR-037
added precomputed aggregates and an enumerated id space. Each was the right answer to a
real invention, and each was verified to fix it. Together they have turned the context
block from *evidence placed in front of the agent* into something closer to *a script the
agent fills in* — it now states which turn is the answer to "which stands out", and
instructs the agent not to re-rank. That is defensible while the failure mode is
invention, but it steadily narrows what the agent contributes beyond formatting, and it
means the quality of an answer increasingly reflects the quality of the Python that built
its context. Worth revisiting if a tool-calling model ever becomes reachable (ADR-030).

**The guards needed guarding.** ADR-027 exists because silent exception handling hid three
live failures; ADR-033 exists because ADR-031's guard blocked correct answers; ADR-039
exists because a guard written for ADR-036 could not fail at all. Three times now, the
mechanism added to catch a class of mistake has itself become an instance of it. The
pattern is consistent enough to expect a fourth: each new guard should be assumed broken
until its failure has been observed, and the cost of that assumption — a mutate/restore
cycle per guard — is the price of the protection being real rather than nominal.
