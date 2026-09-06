// Vocal Stress Co-Pilot dashboard — plain JS, no build step, talks to web/backend.py.
//
// Visual language follows AssemblyAI_InterfaceDemo (Figma Make). Where that demo used
// mock arrays, every visual here is driven by real session evidence: the waveform is
// per-turn arousal, the pitch contour is measured f0_mean, and the pulsing markers are
// turns the system itself flagged unreliable (ADR-025/ADR-026) — not decoration.

const state = { sessions: [], currentSessionId: null, currentSession: null, selectedTurnId: null };

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.detail || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

const scoreClass = (s) => (s >= 0.66 ? "high" : s >= 0.4 ? "mid" : "low");
const scoreColor = (s) => (s >= 0.66 ? "#ff5c5c" : s >= 0.4 ? "#f5a623" : "#4cefb4");
const turnColor = (t) => (t.is_baseline_turn ? "#4a5068" : scoreColor(t.stress.score));
const durationOf = (t) => t.stress.duration_ms ?? t.t_end_ms - t.t_start_ms;

function escapeText(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ── sessions ────────────────────────────────────────────────────────
async function loadSessions() {
  state.sessions = await api("/api/sessions");
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  if (!state.sessions.length) {
    list.innerHTML = `<li class="hint">No sessions yet. Run scripts/build_demo_session.py.</li>`;
    return;
  }
  for (const s of state.sessions) {
    const li = document.createElement("li");
    li.dataset.sessionId = s.session_id;
    li.innerHTML =
      `<span class="sid">${s.session_id}</span>` +
      `<span class="meta">${s.n_turns} TURNS${s.is_synthetic_demo ? " · DEMO" : ""}</span>`;
    li.addEventListener("click", () => selectSession(s.session_id));
    list.appendChild(li);
  }
}

async function selectSession(sessionId) {
  state.currentSessionId = sessionId;
  state.currentSession = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
  state.selectedTurnId = null;

  document.querySelectorAll("#session-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.sessionId === sessionId)
  );

  document.getElementById("empty-state").hidden = true;
  document.getElementById("session-view").hidden = false;
  document.getElementById("detail-body").hidden = true;
  document.getElementById("detail-empty").hidden = false;
  document.getElementById("chat-log").innerHTML =
    `<div class="chat-msg agent chat-intro">Ask why a turn scored the way it did. Every answer
     cites this session's measured evidence — and says so when the evidence cannot support
     an answer.</div>`;
  document.getElementById("chat-suggestions").hidden = false;

  renderHeader();
  renderWaveform();
  renderPitchChart();
  renderTurnList();
  renderAnalytics();
}

function renderHeader() {
  const s = state.currentSession;
  const turns = s.turns;
  const flagged = turns.filter((t) => !t.is_baseline_turn && t.stress.score >= 0.66).length;
  const lowConf = turns.filter((t) => t.stress.score_reliable === false).length;
  const totalMs = turns.reduce((sum, t) => sum + durationOf(t), 0);

  document.getElementById("session-title").textContent = s.session_id;
  document.getElementById("topbar-sub").textContent =
    `${turns.length} TURNS · ${(totalMs / 1000).toFixed(1)}s OF SPEECH ANALYSED`;

  const flaggedPill = document.getElementById("pill-flagged");
  flaggedPill.querySelector("span").textContent = `${flagged} HIGH AROUSAL`;
  flaggedPill.hidden = flagged === 0;

  const lowPill = document.getElementById("pill-lowconf");
  lowPill.querySelector("span").textContent = `${lowConf} LOW CONFIDENCE`;
  lowPill.hidden = lowConf === 0;

  document.getElementById("demo-badge").hidden = !s.is_synthetic_demo;
}

// ── waveform: symmetric bars, height = arousal (demo's waveform, real data) ──
function renderWaveform() {
  const wave = document.getElementById("waveform");
  const turns = state.currentSession.turns;
  wave.innerHTML = "";

  // Each turn gets a small cluster of bars so the strip reads as a waveform rather
  // than a bar chart; the cluster envelope is that turn's score.
  const barsPerTurn = Math.max(2, Math.floor(48 / turns.length));

  turns.forEach((turn) => {
    const color = turnColor(turn);
    const peak = Math.max(0.06, turn.stress.score);
    const dim = turn.stress.score_reliable === false;

    for (let i = 0; i < barsPerTurn; i++) {
      // Envelope: tallest mid-cluster, tapering at the edges.
      const phase = (i + 0.5) / barsPerTurn;
      const envelope = Math.sin(phase * Math.PI);
      const bar = document.createElement("div");
      bar.className = "waveform-bar";
      bar.dataset.turnId = turn.turn_id;
      bar.style.height = `${Math.max(4, peak * envelope * 100)}%`;
      bar.style.background = color;
      bar.style.color = color;
      if (dim) bar.style.opacity = ".3";
      bar.title = `${turn.turn_id} — arousal ${turn.stress.score.toFixed(3)}${
        dim ? " (low confidence)" : ""
      }`;
      bar.addEventListener("click", () => selectTurn(turn.turn_id));
      wave.appendChild(bar);
    }
  });
}

// ── pitch contour: measured f0 per turn, smooth curve, gaps marked ──
function renderPitchChart() {
  const svg = document.getElementById("pitch-chart");
  const axis = document.getElementById("viz-axis");
  const turns = state.currentSession.turns;
  const W = 900, H = 110, PAD_Y = 14;

  axis.innerHTML = turns
    .map((t) => `<span>${t.turn_id.replace(/^t0*/, "T")}</span>`)
    .join("");

  // ADR-026: a 0 Hz "value" is a placeholder, not a measurement — plotting it would
  // draw a cliff that never happened.
  const measured = turns
    .map((t, i) => ({ i, hz: t.prosody_raw.f0_detected ? t.prosody_raw.f0_mean_hz : null }))
    .filter((p) => p.hz && p.hz > 0);

  if (measured.length < 2) {
    svg.innerHTML =
      `<text x="10" y="60" fill="#4a5068" font-size="11" font-family="monospace">` +
      `NOT ENOUGH PITCH MEASUREMENTS IN THIS SESSION</text>`;
    return;
  }

  const hz = measured.map((p) => p.hz);
  const min = Math.min(...hz), max = Math.max(...hz), span = max - min || 1;
  const x = (i) => (i / Math.max(turns.length - 1, 1)) * W;
  const y = (v) => H - PAD_Y - ((v - min) / span) * (H - PAD_Y * 2);

  // Catmull-Rom style smoothing so the line reads like the demo's contour.
  const pts = measured.map((p) => [x(p.i), y(p.hz)]);
  let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[i + 1];
    const cx = (x0 + x1) / 2;
    d += ` C ${cx.toFixed(1)} ${y0.toFixed(1)}, ${cx.toFixed(1)} ${y1.toFixed(1)}, ${x1.toFixed(1)} ${y1.toFixed(1)}`;
  }
  const fillPath = `${d} L ${pts[pts.length - 1][0].toFixed(1)} ${H} L ${pts[0][0].toFixed(1)} ${H} Z`;

  const untracked = turns
    .map((t, i) => ({ i, failed: !t.prosody_raw.f0_detected }))
    .filter((p) => p.failed)
    .map(
      (p) =>
        `<circle cx="${x(p.i).toFixed(1)}" cy="${H - PAD_Y}" r="4" fill="#ff5c5c" class="noise-dot" />`
    )
    .join("");

  const dots = measured
    .map(
      (p) =>
        `<circle cx="${x(p.i).toFixed(1)}" cy="${y(p.hz).toFixed(1)}" r="3" fill="#00d4c8" />`
    )
    .join("");

  svg.innerHTML = `
    <defs>
      <linearGradient id="pitchGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#00d4c8" />
        <stop offset="100%" stop-color="#00d4c800" />
      </linearGradient>
    </defs>
    <path d="${fillPath}" class="pitch-fill" />
    <path d="${d}" class="pitch-line" />
    ${dots}${untracked}
    <text x="4" y="12" fill="#4a5068" font-size="9" font-family="monospace">${max.toFixed(0)}</text>
    <text x="4" y="${H - 4}" fill="#4a5068" font-size="9" font-family="monospace">${min.toFixed(0)}</text>`;
}

// ── turn bubbles ────────────────────────────────────────────────────
function renderTurnList() {
  const list = document.getElementById("turn-list");
  list.innerHTML = "";

  for (const turn of state.currentSession.turns) {
    const color = turnColor(turn);
    const flags = [];
    if (turn.is_baseline_turn) flags.push(`<span class="pill pill-muted">BASELINE</span>`);
    if (turn.stress.score_reliable === false) flags.push(`<span class="pill pill-red">LOW CONF</span>`);
    if (turn.prosody_raw.f0_detected === false) flags.push(`<span class="pill pill-amber">NO PITCH</span>`);

    const el = document.createElement("button");
    el.type = "button";
    el.className = "turn-bubble";
    el.dataset.turnId = turn.turn_id;
    el.addEventListener("click", () => selectTurn(turn.turn_id));
    el.innerHTML = `
      <div class="turn-top">
        <div class="turn-avatar" style="background:${color}1c;border:1px solid ${color}55;color:${color}">
          ${turn.turn_id.replace(/^t0*/, "")}
        </div>
        <span class="turn-id">${turn.turn_id.toUpperCase()}</span>
        <span class="turn-dur">${durationOf(turn)}ms</span>
        <span class="turn-flags">${flags.join("")}</span>
      </div>
      <p class="turn-text ${turn.transcript ? "" : "empty"}">${escapeText(
        turn.transcript || "(no speech transcribed)"
      )}</p>
      <div class="turn-metrics">
        <div class="turn-metric">
          <span>AROUSAL</span>
          <div class="meter-track"><div class="meter-bar" style="width:${(turn.stress.score * 100).toFixed(0)}%;background:${color}"></div></div>
          <span style="color:${color}">${turn.stress.score.toFixed(2)}</span>
        </div>
        ${
          turn.prosody_raw.f0_detected
            ? `<div class="turn-metric"><span>f0</span><span style="color:var(--text-secondary)">${turn.prosody_raw.f0_mean_hz.toFixed(0)} Hz</span></div>`
            : ""
        }
      </div>`;
    list.appendChild(el);
  }
}

// ── analytics tab ───────────────────────────────────────────────────
function renderAnalytics() {
  const turns = state.currentSession.turns;

  document.getElementById("dist-bars").innerHTML = turns
    .map((t) => {
      const color = turnColor(t);
      return `
        <div class="dist-row">
          <span class="dist-name">${t.turn_id.replace(/^t0*/, "T")}</span>
          <div class="dist-track"><div class="dist-fill" style="width:${(t.stress.score * 100).toFixed(0)}%;background:${color}"></div></div>
          <span class="dist-val" style="color:${color}">${t.stress.score.toFixed(2)}</span>
        </div>`;
    })
    .join("");

  // "Events" here are the system's own admissions of unreliability, not detections.
  const events = [];
  for (const t of turns) {
    if (t.stress.score_reliable === false)
      events.push({ turn: t.turn_id, color: "#ff5c5c", what: `${durationOf(t)}ms — below floor` });
    if (t.prosody_raw.f0_detected === false)
      events.push({ turn: t.turn_id, color: "#f5a623", what: "pitch tracking failed" });
  }
  document.getElementById("reliability-events").innerHTML = events.length
    ? events
        .map(
          (e) => `
        <div class="event-row">
          <i class="dot noise-dot" style="background:${e.color}"></i>
          <span class="mono">${e.turn.toUpperCase()}</span>
          <span class="event-what">${e.what}</span>
        </div>`
        )
        .join("")
    : `<div class="event-empty">No reliability issues in this session.</div>`;

  const scored = turns.filter((t) => !t.is_baseline_turn);
  const mean = scored.length ? scored.reduce((s, t) => s + t.stress.score, 0) / scored.length : 0;
  const reliablePct = turns.length
    ? (turns.filter((t) => t.stress.score_reliable !== false).length / turns.length) * 100
    : 0;
  const pitchPct = turns.length
    ? (turns.filter((t) => t.prosody_raw.f0_detected).length / turns.length) * 100
    : 0;

  document.getElementById("session-markers").innerHTML = [
    { label: "Mean arousal (scored turns)", val: mean.toFixed(2), pct: mean * 100, color: scoreColor(mean) },
    { label: "Turns above reliability floor", val: `${reliablePct.toFixed(0)}%`, pct: reliablePct, color: "#00d4c8" },
    { label: "Turns with measured pitch", val: `${pitchPct.toFixed(0)}%`, pct: pitchPct, color: "#8b78ff" },
  ]
    .map(
      (m) => `
      <div>
        <div class="meter-row-head">
          <span class="meter-name">${m.label}</span>
          <span class="meter-val" style="color:${m.color}">${m.val}</span>
        </div>
        <div class="meter-track"><div class="meter-bar" style="width:${m.pct.toFixed(0)}%;background:${m.color}"></div></div>
      </div>`
    )
    .join("");
}

// ── right panel ─────────────────────────────────────────────────────
function selectTurn(turnId) {
  state.selectedTurnId = turnId;
  const turn = state.currentSession.turns.find((t) => t.turn_id === turnId);
  if (!turn) return;

  document.querySelectorAll(".turn-bubble").forEach((el) =>
    el.classList.toggle("selected", el.dataset.turnId === turnId)
  );
  document.querySelectorAll(".waveform-bar").forEach((el) =>
    el.classList.toggle("selected", el.dataset.turnId === turnId)
  );

  document.getElementById("detail-empty").hidden = true;
  document.getElementById("detail-body").hidden = false;

  const cls = scoreClass(turn.stress.score);
  document.getElementById("detail-turn-id").textContent =
    `${turn.turn_id.toUpperCase()} · ${turn.is_baseline_turn ? "CALIBRATION" : "SCORED"} · ${durationOf(turn)}ms`;
  const scoreEl = document.getElementById("detail-score");
  scoreEl.textContent = turn.stress.score.toFixed(3);
  scoreEl.parentElement.className = `hero-number ${cls}`;

  const badge = document.getElementById("detail-badge");
  badge.textContent = turn.stress.label.toUpperCase();
  badge.className = `pill pill-${cls === "high" ? "red" : cls === "mid" ? "amber" : "green"}`;

  renderMeters(turn);

  // Alerts: the system stating when its own number should not be trusted.
  const alerts = [];
  if (turn.stress.score_reliable === false)
    alerts.push(`<div class="alert alert-red"><i class="dot dot-red noise-dot"></i>BELOW RELIABILITY FLOOR — SCORE IS LOW-CONFIDENCE</div>`);
  if (turn.prosody_raw.f0_detected === false)
    alerts.push(`<div class="alert alert-amber"><i class="dot dot-amber"></i>PITCH TRACKING FAILED — PITCH METRICS WITHHELD</div>`);
  if (!turn.stress.calibrated)
    alerts.push(`<div class="alert alert-muted"><i class="dot" style="background:#4a5068"></i>NO RELIABLE BASELINE FOR THIS SESSION</div>`);
  document.getElementById("detail-alerts").innerHTML = alerts.join("");

  document.getElementById("detail-transcript").textContent =
    turn.transcript || "(no speech transcribed)";
  document.getElementById("detail-gradcam-text").textContent =
    turn.xai.gradcam_description || "No attention summary recorded for this turn.";
  document.getElementById("detail-disclaimer").textContent = turn.disclaimer;

  const artifactUrl = (kind) =>
    `/api/sessions/${encodeURIComponent(state.currentSessionId)}/artifact/${encodeURIComponent(turnId)}/${kind}`;
  setArtifact("gradcam-img", turn.xai.gradcam_png, artifactUrl("gradcam"));
  setArtifact("spec-img", turn.xai.spectrogram_png, artifactUrl("spectrogram"));
}

function setArtifact(id, hasPath, url) {
  const img = document.getElementById(id);
  img.src = hasPath ? url : "";
  img.parentElement.hidden = !hasPath;
}

function renderMeters(turn) {
  const container = document.getElementById("detail-meters");
  const entries = Object.entries(turn.baseline_deviation || {});

  if (!entries.length) {
    container.innerHTML = `<div class="meter-name">No baseline deviations available for this turn.</div>`;
    return;
  }

  // z-scores are signed: the bar shows magnitude, the number keeps the sign.
  container.innerHTML = entries
    .map(([name, z]) => {
      const magnitude = Math.min(Math.abs(z) / 3, 1) * 100;
      const color = Math.abs(z) >= 2 ? "#ff5c5c" : Math.abs(z) >= 1 ? "#f5a623" : "#00d4c8";
      return `
        <div>
          <div class="meter-row-head">
            <span class="meter-name">${name}</span>
            <span class="meter-val" style="color:${color}">${z >= 0 ? "+" : ""}${z.toFixed(2)} SD</span>
          </div>
          <div class="meter-track"><div class="meter-bar" style="width:${magnitude.toFixed(0)}%;background:${color}"></div></div>
        </div>`;
    })
    .join("");
}

// ── tabs ────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("tab-active", t === tab));
    document.getElementById("tab-turns").hidden = tab.dataset.tab !== "turns";
    document.getElementById("tab-analytics").hidden = tab.dataset.tab !== "analytics";
  });
});

// ── analyst chat ────────────────────────────────────────────────────
function appendChatMessage(role, text) {
  const log = document.getElementById("chat-log");
  log.querySelector(".chat-intro")?.remove();
  document.getElementById("chat-suggestions").hidden = true;
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  if (role === "agent") {
    // The agent replies in Markdown; renderMarkdown escapes HTML before formatting, so
    // LLM output (downstream of untrusted candidate speech) cannot inject live markup.
    div.innerHTML = renderMarkdown(text);
  } else {
    div.textContent = text;
  }
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

document.getElementById("chat-suggestions").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  const input = document.getElementById("chat-input");
  input.value = chip.textContent.trim();
  // requestSubmit (not submit) so the form's own handler runs, keeping one code path.
  document.getElementById("chat-form").requestSubmit();
});

document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;

  if (!state.currentSessionId) {
    appendChatMessage("error", "Select a session first — answers are grounded in one session's evidence.");
    return;
  }

  appendChatMessage("user", message);
  input.value = "";
  const button = e.target.querySelector("button");
  button.disabled = true;
  const pending = appendChatMessage("pending", "ANALYSING SESSION EVIDENCE…");

  try {
    const { answer } = await api(
      `/api/sessions/${encodeURIComponent(state.currentSessionId)}/chat`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      }
    );
    pending.remove();
    appendChatMessage("agent", answer);
  } catch (err) {
    pending.remove();
    let msg;
    if (err.status === 503) {
      msg = "Analyst Agent isn't configured — set ASSEMBLYAI_API_KEY server-side (see .env.example).";
    } else if (err.status === 429) {
      msg = err.message; // backend composes an actionable rate-limit message
    } else if (err.status === 502) {
      msg = `The LLM Gateway rejected the request. ${err.message}`;
    } else {
      msg = `Error: ${err.message}`;
    }
    appendChatMessage("error", msg);
  } finally {
    button.disabled = false;
  }
});

loadSessions()
  .then(() => {
    // Deep-link support (ADR-041): capture.html lands here with `?session=<id>` the
    // moment a live session is saved, so "record -> end -> see the analysis" is one
    // click, not a session picked out of a list from memory.
    const wanted = new URLSearchParams(location.search).get("session");
    if (wanted && state.sessions.some((s) => s.session_id === wanted)) {
      selectSession(wanted);
    }
  })
  .catch((err) => {
    document.getElementById("empty-state").textContent = `Failed to load sessions: ${err.message}`;
  });
