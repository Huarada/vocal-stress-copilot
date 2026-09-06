"use strict";

// ADR-041: browser-based capture, replacing scripts/run_interview.py's terminal-only
// microphone path for anyone who doesn't want to run local Python. Talks to
// /ws/interview (web/live_capture.py), which wraps the same LiveInterviewRunner the
// terminal script uses — same turn-finalization guarantees, same evidence pipeline.
//
// UNVERIFIED end to end (2026-09-05): every piece here has been unit-tested at its own
// boundary (the worklet's resampling math, the websocket relay's framing), but the
// full loop — real mic -> AudioWorklet -> WebSocket -> real AssemblyAI Voice Agent ->
// audible reply — has never run, because that requires a human with a real
// microphone and speakers, the same boundary ARCHITECTURE.md already names for the
// terminal script. Run a real session before demoing this.

const REQUIRED_RATE = 24_000;

const state = {
  ws: null,
  audioContext: null,
  mediaStream: null,
  workletNode: null,
  sessionId: null,
  playbackCursor: 0,
  turns: [],
};

const scoreColor = (s) => (s >= 0.66 ? "#ff5c5c" : s >= 0.4 ? "#f5a623" : "#4cefb4");

function $(id) {
  return document.getElementById(id);
}

function showPanel(name) {
  $("capture-idle").hidden = name !== "idle";
  $("capture-live").hidden = name !== "live";
  $("capture-done").hidden = name !== "done";
}

const MAX_STATUS_LINES = 300; // ADR-043: defense in depth, see below

function logStatus(message) {
  const el = $("status-log");
  const line = document.createElement("div");
  // Defense in depth against ADR-043's freeze: the backend now truncates every status
  // message and no longer dumps a raw reply.audio event, but a browser tab has no
  // business trusting a server to always behave — cap both the length of any one line
  // and how many accumulate, so a bug on either side degrades to "log stops scrolling"
  // rather than "tab stops responding."
  const MAX_LINE_CHARS = 600;
  line.textContent =
    message.length > MAX_LINE_CHARS ? `${message.slice(0, MAX_LINE_CHARS)}… [truncated]` : message;
  el.appendChild(line);
  while (el.childElementCount > MAX_STATUS_LINES) {
    el.removeChild(el.firstChild);
  }
  el.scrollTop = el.scrollHeight;
}

function showError(message) {
  const el = $("capture-error");
  el.textContent = message;
  el.hidden = false;
  logStatus(`[error] ${message}`);
}

function appendConversation(role, text) {
  const log = $("conversation-log");
  const div = document.createElement("div");
  div.className = `capture-msg ${role}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function appendTurn(turn) {
  state.turns.push(turn);
  const list = $("live-turn-list");
  const color = turn.is_baseline_turn ? "#4a5068" : scoreColor(turn.stress.score);
  const card = document.createElement("div");
  card.className = "live-turn-card";
  card.style.borderLeftColor = color;
  const flags = [];
  if (turn.is_baseline_turn) flags.push("BASELINE");
  if (turn.stress.score_reliable === false) flags.push("LOW CONF");
  if (turn.prosody_raw.f0_detected === false) flags.push("NO PITCH");
  card.innerHTML =
    `<div class="live-turn-head">` +
    `<span class="mono">${turn.turn_id}</span>` +
    `<span class="mono" style="color:${color}">${turn.stress.score.toFixed(2)}</span>` +
    `</div>` +
    (flags.length ? `<div class="live-turn-flags mono">${flags.join(" · ")}</div>` : "") +
    `<div class="live-turn-transcript">${escapeText(turn.transcript || "(no transcript)")}</div>`;
  list.appendChild(card);
  list.scrollTop = list.scrollHeight;

  // The candidate's own words surface here — once their turn finalizes, not live
  // word-by-word (no separate candidate-transcript hook is wired yet; see ADR-041's
  // "left out" list). Folding it into the same conversation log keeps the page
  // readable as one timeline instead of two out-of-sync ones.
  if (turn.transcript) {
    appendConversation("candidate", turn.transcript);
  }
}

function escapeText(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function updateMicLevel(int16Chunk) {
  let sumSquares = 0;
  for (let i = 0; i < int16Chunk.length; i++) {
    const s = int16Chunk[i] / 32768;
    sumSquares += s * s;
  }
  const rms = Math.sqrt(sumSquares / int16Chunk.length);
  const pct = Math.min(100, Math.round(rms * 400)); // headroom: full-scale speech rarely hits RMS 0.25
  $("mic-level-bar").style.width = `${pct}%`;
}

// --- agent reply playback: raw PCM16 frames scheduled back-to-back on one timeline --

function playAgentAudio(arrayBuffer) {
  const ctx = state.audioContext;
  if (!ctx) return;
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / (int16[i] < 0 ? 32768 : 32767);
  }
  const buffer = ctx.createBuffer(1, float32.length, REQUIRED_RATE);
  buffer.copyToChannel(float32, 0);

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);

  // Schedule immediately after whatever's already queued, not at "now" — back-to-back
  // chunks played each at ctx.currentTime would overlap/garble; a running cursor gives
  // gapless playback for a stream of small chunks.
  const startAt = Math.max(ctx.currentTime, state.playbackCursor);
  source.start(startAt);
  state.playbackCursor = startAt + buffer.duration;
}

// --- websocket -----------------------------------------------------------------

function connectSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/interview`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "handshake", sample_rate_hz: REQUIRED_RATE }));
    logStatus("[ws] connected, handshake sent");
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleServerMessage(JSON.parse(event.data));
    } else {
      playAgentAudio(event.data);
    }
  };

  ws.onerror = () => {
    logStatus("[ws] connection error");
  };

  ws.onclose = (event) => {
    logStatus(`[ws] closed (code=${event.code})`);
    teardownAudio();
  };
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "session_started":
      state.sessionId = msg.session_id;
      $("session-id-badge").textContent = msg.session_id;
      logStatus(`[session] started ${msg.session_id}`);
      break;
    case "status":
      logStatus(msg.message);
      break;
    case "agent_text":
      appendConversation("agent", msg.text);
      break;
    case "turn":
      appendTurn(msg.turn);
      break;
    case "error":
      showError(msg.message);
      break;
    case "session_saved":
      onSessionSaved(msg.session_id);
      break;
    default:
      logStatus(`[ws] unhandled message type ${JSON.stringify(msg.type)}`);
  }
}

function onSessionSaved(sessionId) {
  $("done-summary").textContent =
    `${state.turns.length} turn${state.turns.length === 1 ? "" : "s"} recorded in session ${sessionId}.`;
  $("view-analysis-link").href = `index.html?session=${encodeURIComponent(sessionId)}`;
  showPanel("done");
}

// --- lifecycle -------------------------------------------------------------------

async function startSession() {
  $("capture-error").hidden = true;

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: REQUIRED_RATE,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (err) {
    showError(`Microphone access denied or unavailable: ${err.message}`);
    return;
  }
  state.mediaStream = stream;

  // ASSUMPTION: requesting `sampleRate` on the AudioContext constructor is honoured on
  // Chrome/Firefox desktop but is not guaranteed everywhere (Safari in particular can
  // silently ignore it and hand back the hardware rate). SKILLS.md Hard Rule 5: assert,
  // don't assume — capture-worklet.js resamples to REQUIRED_RATE regardless of what
  // the context actually grants, and the handshake below declares the rate actually
  // being sent, not the one requested here.
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextCtor({ sampleRate: REQUIRED_RATE });
  state.audioContext = audioContext;
  logStatus(`[mic] AudioContext running at ${audioContext.sampleRate}Hz (requested ${REQUIRED_RATE}Hz)`);

  try {
    await audioContext.audioWorklet.addModule("capture-worklet.js?v=6");
  } catch (err) {
    showError(`This browser does not support AudioWorklet: ${err.message}`);
    return;
  }

  const source = audioContext.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(audioContext, "pcm-capture-processor", {
    processorOptions: {
      inputSampleRate: audioContext.sampleRate,
      targetSampleRate: REQUIRED_RATE,
    },
  });
  state.workletNode = worklet;
  source.connect(worklet);
  // Deliberately not connected to `audioContext.destination` — this node observes the
  // mic, it does not play it back (that would be an audible echo of the candidate's
  // own voice).

  worklet.port.onmessage = (event) => {
    const int16Chunk = event.data; // Int16Array at REQUIRED_RATE, from the worklet
    updateMicLevel(int16Chunk);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(int16Chunk.buffer);
    }
  };

  connectSocket();
  showPanel("live");
}

function endSession() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "end_session" }));
  }
  teardownAudio();
}

function teardownAudio() {
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }
  if (state.workletNode) {
    state.workletNode.port.onmessage = null;
    state.workletNode.disconnect();
    state.workletNode = null;
  }
  // audioContext is left open until session_saved arrives, so any final reply.audio
  // frames still in flight can still be scheduled and played.
}

$("start-btn").addEventListener("click", startSession);
$("end-btn").addEventListener("click", endSession);
