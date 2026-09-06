# SKILLS.md — how to work on this project

Operating guide for Claude and for Lucas. Read alongside [ARCHITECTURE.md](ARCHITECTURE.md),
which owns the *what*, and [ADR.md](ADR.md), which owns the *why* (every decision, its
trade-offs, and the full bug catalogue). This file owns the *how* — which competencies the
work actually needs, which facts are already settled, and which mistakes are expensive here.

---

## 1. Skill matrix

Ranked by how much the project's outcome depends on them.

| # | Skill | Where it bites | Status |
|---|-------|----------------|--------|
| 1 | **Real-time audio plumbing** — WebSocket streaming, ring buffers, sample-rate discipline, clock alignment | Ingest, Sync Bus. The single largest source of subtle bugs in this system | 🔴 New — no prior code |
| 2 | **Speech DSP / prosody** — narrowband vs wideband spectrograms, F0, jitter, shimmer, HNR | Acoustic Service, XAI. Determines whether explanations are meaningful or decorative | 🟡 Spectrogram side solved; descriptor side new |
| 3 | **TF/Keras transfer learning + honest evaluation** — grouped CV, class weights, cross-corpus generalization, probability calibration | Model track. Determines whether any number reported is real | 🟢 Strong — existing pipeline does this well |
| 4 | **LLM tool-calling design** — JSON Schema tools, grounding responses in tool results, refusing ungrounded claims | Analyst Agent. This *is* the XAI product | 🟡 New API, familiar concept |
| 5 | **XAI** — Grad-CAM on CNNs, feature attribution, communicating uncertainty | XAI layer. The project's actual differentiator | 🟡 Grad-CAM is cheap given the CNN-on-image design |
| 6 | **Demo/presentation craft** — video + slides, dashboard legibility, narrative | One of 4 equally-listed judging criteria (Application of Technology / Presentation / Business Value / Originality — official rubric, no percentages given; corrected 2026-09-06, "25%" below was a pre-rubric guess), routinely underinvested | 🔴 Not started |
| 7 | **Research ethics & licensing hygiene** | README, `.gitignore`, consent gate | 🟢 Already reasoned through |

### Where to spend the marginal hour

Not on model accuracy. The acoustic model is the part that already works, and squeezing
another 3 points out of it changes nothing about how the project is scored. The marginal hour
goes to **#1** (because it silently corrupts everything downstream if wrong) and **#6**
(because "Presentation" is a full quarter of the rubric and a working system that demos badly
loses to a simpler one that demos well).

---

## 2. Hard rules

Violating these is a bug, not a style preference.

1. **Never claim deception.** No output, prompt, variable name, UI string, or slide says
   "lying", "deceptive", "guilty", or "truthful". The vocabulary is *vocal stress*,
   *arousal*, *deviation from baseline*, *confidence*. This is enforced in
   `agents/prompts/analyst.md` and must be re-checked whenever that prompt changes.
2. **The Interview Agent never sees a stress score.** No arrow from evidence back into the
   conversation (ARCHITECTURE.md §2). If a feature seems to need one, it's the wrong feature.
3. **Never commit datasets, derived images, or trained weights.** RAVDESS is CC BY-NC-SA;
   MUStARD++ has its own terms. Code is publishable; their data and anything derived from it
   is not. Check `.gitignore` before the first commit, not after.
4. **Group by speaker in every split.** `StratifiedGroupKFold` grouped by `Actor_XX` for
   RAVDESS. An ungrouped split leaks speaker identity and produces a beautiful, meaningless
   accuracy.
5. **Assert sample rate at every boundary.** The model was trained at 16 kHz; the Voice Agent
   path runs at 24 kHz. Any function that receives audio asserts its rate rather than assuming.
6. **Report the cross-corpus number.** In-corpus OOF is for development. RAVDESS → MUStARD++
   `Arousal` is the number that goes in the README and the video.
7. **Mutation-test every guard protecting a catalogued bug** (ADR-039). A test that has
   never been observed to fail is not coverage — it is a claim of coverage. Break the
   invariant, watch *that* test go red, restore, diff against the backup. One of the eight
   frontend guards was green and useless; only mutation revealed it. Grep generated test
   files for control characters in the same pass: a `\b` in a non-raw Python string is
   written to disk as a backspace byte and silently weakens the regex around it.
8. **State facts the model would otherwise derive** (ADR-034, ADR-037). Never leave the
   agent to rank, count, or group rows of context prose. It reported the session maximum as
   0.958 when it was 0.997, and cited turn ids that did not exist. Precompute in Python,
   enumerate the valid id space, and state set *membership* rather than set size — counts
   alone still left it grouping by eye.

---

## 3. Settled facts — do not re-derive these

### Existing model (`../sarcasmoVoz/DetectarSarcasmoPorVoz-main/`)
- TensorFlow/Keras. Checkpoint on disk: `resnet1_prelayer1.keras` — **this is a
  32×32×3 CIFAR-10 pretraining checkpoint, not a spectrogram-domain sarcasm model.**
  The 96×128 MUStARD++ training loop (notebook cell 20) builds a fresh model per CV fold
  and never calls `.save()`; no spectrogram-trained checkpoint exists on disk. It is
  still usable as a warm-start (the architecture is fully convolutional + GAP, so weight
  shapes don't depend on input H×W — confirmed empirically: 33/34 layers transplant), but
  treat the transfer as an **unproven A/B experiment** (implemented: it is, in
  `scripts/train_arousal_model.py`), not an assumed win — the domain gap is real
  (natural photos vs. audio spectrograms).
- `build_resnet_light()`: 8→16 filters, 2 residual blocks, `SpatialDropout2D(0.25)`,
  GAP → BatchNorm → `Dropout(0.5)` → softmax(2), Adam,
  `CategoricalCrossentropy(label_smoothing=0.1)`. **6,578 weight params**
  (`model.count_params()` — not 19,384; that figure from `.summary()` includes Adam's
  optimizer-state slots).
- Input: **96×128×3**, values in [0,1]. **Not literal RGB** — R = log-mel spectrogram
  (96 mel bins), G = first-order delta-MFCC, B = second-order delta-MFCC, each
  independently z-scored/clipped (`_zscore_global`). Porting only the log-mel channel
  and repeating it 3x (a plausible-looking shortcut) would throw away the
  delta/delta-delta channels the original model was actually trained to read.
- Preprocessing: 16 kHz mono → narrowband spectrogram (**80 ms window, 10 ms hop**) →
  per-channel global z-score, clip [-3,3], map to [0,1] → resize to 96×128. Includes a
  contrast guard (raises below 3 dB log-mel dynamic range — the original's "tapete
  cinza"/gray-carpet check) — ported as `FlatSpectrogramError` in
  `infrastructure/audio/spectrogram.py`.
- Training: `StratifiedGroupKFold(5)`, `compute_class_weight('balanced')`,
  `ReduceLROnPlateau` + `LearningRateScheduler` + `EarlyStopping(patience=15)`, batch 16
- Reference notebook: `DetectarSarcasmoDataAgumentationMustardPlusPrecisaoFscore.ipynb`
  (cells 11–14 = augmentation + spectrogram; cell 17–20 = model + CV)
- **Ported and implemented** in `src/voicestress/infrastructure/{audio,models}/` — see
  ARCHITECTURE.md §7a for what's built vs. not, and §8 for the file-by-file map.

### Data on disk
- `../databaseAudio/audioSpeech/Actor_01..24/` — RAVDESS speech, **1440 files** (use this)
- `../databaseAudio/audioSong/` — RAVDESS song, 1012 files (out-of-domain; not for training)
- `../sarcasmoVoz/.../audio_extracted_16k/` — MUStARD++ at 16 kHz, with
  `MUStARD_plusplus_with_audio_16k.csv` carrying `Sarcasm`, `Valence`, **`Arousal`**,
  `Implicit_Emotion`, `Explicit_Emotion` → this is the cross-corpus test set

### RAVDESS filename encoding
`modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav`
Emotion: `01` neutral · `02` calm · `03` happy · `04` sad · `05` angry · `06` fearful ·
`07` disgust · `08` surprised. Odd actor id = male, even = female.
Arousal binarization and the exclusion of `03 happy`: ARCHITECTURE.md §6.

### AssemblyAI (verified against docs 2026-09-04)
- Voice Agent: `wss://agents.assemblyai.com/v1/ws`, PCM16 **24 kHz**, base64 in JSON.
  Client→server `session.update` / `input.audio` / `tool.result` / `reply.create`;
  server→client `session.ready` / `input.speech.started|stopped` / `transcript.user` /
  `reply.started|audio|done` / `tool.call`
- Tools: `session.tools[]` with `type:"function"`, `parameters` as JSON Schema,
  `execution_mode` (confirmed 2026-09-06 against AssemblyAI's own tool-calling docs,
  ADR-045) is **per-tool, latency-scoped, not a content-review hook**:
  `"interactive"` (default; short tools, <~5s — agent speaks a transition phrase like
  "let me check" and keeps going) vs `"hold"` (long-running tools, >~10s — agent stays
  silent and ignores user-speech-triggered replies until `tool.result` arrives, then
  speaks the result directly). Neither mode gates or reviews the agent's *own generated
  reply* before it's spoken — `reply.audio` streams incrementally as the LLM generates
  ("you don't wait for the full reply to start playing," per AssemblyAI's own 5-minute
  walkthrough), so there is no documented interception point between text generation
  and audio playback. This is why the Analyst Agent was NOT moved onto a second Voice
  Agent connection despite ADR-044 confirming tool-calling works there — its vocabulary
  guard (Hard Rule 1) depends on inspecting text before it reaches anyone, which this
  protocol has no hook for.
- Streaming STT: `wss://streaming.assemblyai.com/v3/ws`, 16 kHz PCM, word-level timestamps,
  **billed by connection duration** — close sockets
- LLM Gateway: OpenAI-compatible, tool calling + structured outputs

### Environment
- Windows 11, PowerShell primary (Bash tool available). Paths under OneDrive — expect
  occasional file locking; don't write large artifacts into synced folders if avoidable.
- TF GPU on this machine is DirectML-flavored (the notebooks probe `is_built_with_cuda`
  and `is_built_with_rocm` and set `set_memory_growth`). Don't assume CUDA.

---

## 4. Decision heuristics

**When the model and the integration compete for time, the integration wins.** A mediocre
score wired end-to-end through a live voice agent demonstrates the hackathon's actual theme;
a great score in a notebook demonstrates nothing the judges are scoring.

**When a number looks too good, suspect the split.** 90%+ on this task means speaker leakage,
`audioSong` contamination, or `happy` doing the work. Check grouping first, every time.

**When an explanation sounds confident, check it against a tool result.** The Analyst Agent
should be *boring and grounded*. Fluent speculation is the failure mode that would make this
project exactly the pseudoscience it's trying not to be.

**When scope must be cut, follow ARCHITECTURE.md §9** — and never cut baseline calibration,
the disclaimer vocabulary, or the cross-corpus evaluation. They're cheap and they're the
credibility.

---

## 5. Anti-patterns seen in this problem space

- **Spectrogram-as-image without checking what the image contains.** Verify contrast and
  harmonic structure on a few samples before training on 1440 of them. The existing notebook
  already has a "mel sem contraste" guard — keep that habit.
- **Treating acted emotion as real stress.** RAVDESS actors perform fear; a nervous candidate
  is not performing. Baseline-relative scoring narrows the claim enough to survive this; a raw
  absolute score does not.
- **A dashboard that shows a number without its uncertainty.** A bare "0.62" invites exactly
  the over-reading the whole design is built to prevent. Show the confidence band and the
  baseline it's relative to, always.
- **Letting the LLM name the cause.** It explains *which acoustic features deviated*. Why a
  human's voice deviated — stress, fatigue, a bad connection, personality — is not
  determinable from audio, and the prompt must say so.

---

## 6a. Bugs the concept→implementation→validation loop actually caught

Kept here because each one is a *class* of mistake this codebase is now guarded against
by a regression test — read before "simplifying" the code that guards them:

1. **Keras layer auto-naming is process-global, not per-model.** Matching checkpoint
   layers by `.name` breaks from the 2nd `build_resnet_light()` call onward in any long-
   running process (a k-fold loop builds 5+ models) — it silently transplants 0 layers
   while still claiming "warm-started". Fixed by matching positionally by layer type.
   Guard: `tests/integration/test_checkpoint_transplant.py`.
2. **String-suffix-stripping to derive dict keys drifts from the documented contract.**
   `BaselineProfile.deviation()` originally built keys via
   `name.replace('_hz','').replace('_pct','')...`, which produced `jitter_local_z`
   instead of the documented `jitter_z`. Fixed with an explicit field→key table. Guard:
   `tests/unit/test_baseline_profile.py::test_deviation_sign_and_magnitude`.
3. **`@dataclass(slots=True)` rejects attributes set in `__post_init__`** that aren't
   declared constructor fields (e.g. a lazily-built Keras sub-model). Use plain
   `@dataclass` for anything that attaches derived state after construction; reserve
   `slots=True` for the immutable domain value objects it was chosen for.
4. **A test fixture that varies only one of six correlated features can trip a
   correctness guard that's working as intended**, not the code under test — a
   `BaselineProfile` calibrated from turns that only vary F0 legitimately has zero
   variance on the other five features, and `is_reliable` is *supposed* to reject that.
   When a test fails, check whether the fixture or the guard is unrealistic before
   "fixing" the guard.
5. **`resnet1_prelayer1.keras` is a 32×32 CIFAR-10 checkpoint, not a spectrogram-domain
   one** — see §3 above. Caught by loading it directly and reading `model.summary()`
   during implementation, not by trusting the notebook's variable names.
6. **A test can be green and protect nothing.** A frontend guard searched `app.js` for
   `turn.score` while the drift it existed to catch produces `t.score` — eight tests
   written, seven real, and only deliberate mutation told them apart. Never let "it
   passed" stand as evidence a guard works (ADR-039).
7. **`\b` inside a non-raw Python string is a backspace byte, not a word boundary.**
   Generating test files from a script silently wrote `\b` into a regex and weakened
   it; `pytest` stayed green. Use raw strings for generated content, and grep the written
   file for control characters.
8. **The model cannot rank or count rows of context prose.** Given fourteen turns it
   reported the session maximum as 0.958 when it was 0.997, and invented turn ids
   `t0039`/`t0040` for a session numbered t0001–t0014. Even given correct *counts* it
   still grouped by eye, calling a 2600 ms turn low-confidence. Precompute aggregates,
   enumerate the id space, and state set membership (ADR-037).
9. **A prominent feature is a discovered feature.** The session-scope invention above had
   existed for as long as context grounding had, and surfaced within minutes of putting
   three starter prompts one click away. Ease of invocation is a debugging tool, not only
   a UX property (ADR-038).
10. **A cancellation must land inside the same `try/finally` it needs to trigger.**
    `LiveInterviewRunner.run()`'s save-on-exit logic sat one indent too shallow —
    around the event loop, not the whole method — so a cancellation during `configure()`
    skipped it entirely while the caller still reported success. Found on the first real
    connection with a live key, not by any of 270 tests using a fake one (ADR-042).
11. **An unverified guess about a payload's shape must degrade, not flood.** Guessing
    `reply.audio`'s key wrong meant every chunk hit a fallback that logged the *entire*
    raw event — a multi-kilobyte base64 blob — as one line, freezing the browser tab
    within a few real exchanges. Try several plausible keys instead of one, and make the
    no-match diagnostic report shapes and names only, never values (ADR-043).

## 6b. Running it

```bash
pip install -e .                                    # editable install, once
python scripts/build_features.py --corpus ravdess    # ~30s, 1248 clips, 0 failures expected
python scripts/build_features.py --corpus mustard    # ~10s, 504 clips, 0 failures expected
python scripts/train_arousal_model.py                # A/B CV + final model + cross-corpus eval
                                                       # writes artifacts/metrics/latest.json

python -m pytest tests/ -m "not quality_gate"         # unit+integration, no training required, ~10s
python -m pytest tests/quality_gates/                 # reads latest.json, enforces §6 thresholds
python -m pytest tests/                               # everything — 287 tests as of this pass, ~21s

python scripts/build_demo_session.py                  # real model + real audio → artifacts/sessions/
python -m uvicorn web.backend:app --reload             # dashboard at http://localhost:8000
# ^ keep --reload. Without it, uvicorn holds imported Python modules in memory and
#   backend edits (guards, evidence rendering, prompts) silently keep running the OLD
#   code — this cost real debugging time once. Static JS/CSS is already protected by the
#   no-store middleware, but Python needs the restart.

# Live agent, terminal mic path — needs ASSEMBLYAI_API_KEY in .env and a working mic:
python scripts/run_interview.py

# Live agent, browser capture path (ADR-041) — same API key requirement, no local mic
# drivers needed. Full real-mic-to-real-agent loop NOT yet live-tested (ARCHITECTURE.md
# §7b) — start uvicorn above, then open http://localhost:8000/capture.html.
```

TF prints oneDNN/CPU-feature banners to stderr on every run — harmless, pipe through
`grep -viE "oneDNN|cuda|cuFFT|cuDNN|TF-TRT|cpu_feature|absl|rebuild TensorFlow"` if it's
cluttering output you're reading closely.

## 7. For Claude specifically

- Prefer reusing functions from the sarcasm notebooks verbatim over rewriting them. They are
  tested; a "cleaner" reimplementation is a new source of preprocessing drift.
- When touching anything under `service/`, state which side of the 16/24 kHz boundary the code
  is on before writing it.
- Don't propose training from scratch. There is a warm-start checkpoint and 26 days.
- When asked for accuracy numbers, ask which split produced them before reporting them.
- Portuguese for conversation with Lucas; English for repo artifacts, code, and anything a
  judge might read.
