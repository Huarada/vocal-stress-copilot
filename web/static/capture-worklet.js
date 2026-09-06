/**
 * Runs on the browser's real-time audio thread (AudioWorkletGlobalScope), NOT the main
 * thread — it cannot touch the DOM or the WebSocket directly, only post messages back.
 *
 * Receives 128-sample float32 blocks at the AudioContext's actual rate (`sampleRate`,
 * a global in this scope) and posts ~100ms int16 PCM chunks to the main thread, ready
 * to send over the wire unchanged.
 *
 * SKILLS.md Hard Rule 5 ("assert sample rate at every boundary") is why this resamples
 * at all: capture.js requests a 24kHz AudioContext, but not every browser honours that
 * request (Safari in particular can silently hand back the hardware rate instead). If
 * `inputSampleRate` differs from `targetSampleRate`, every chunk is linearly
 * interpolated down (or up) to the target before it ever reaches the network — the
 * server-side handshake in web/live_capture.py then independently verifies the
 * declared rate rather than trusting it, so a browser that lied here would still be
 * caught, not silently mis-processed.
 *
 * UNVERIFIED (2026-09-05): this class has never run against a real getUserMedia
 * stream — AudioWorkletGlobalScope isn't reachable from a non-browser tool call, the
 * same boundary ARCHITECTURE.md already names for scripts/run_interview.py's mic path.
 * `node --check` confirms this parses; it does not confirm the resampling math sounds
 * correct to a human ear. Run a real session and listen before trusting this comment.
 */
class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.inputRate = opts.inputSampleRate || sampleRate;
    this.targetRate = opts.targetSampleRate || 24000;
    this.ratio = this.inputRate / this.targetRate;
    this.chunkSamples = Math.round(this.targetRate * 0.1); // ~100ms per outgoing chunk
    this.outBuffer = [];
    this._pending = new Float32Array(0); // input samples not yet consumed by resampling
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0] || input[0].length === 0) {
      return true; // no mic data this render quantum — keep the node alive regardless
    }
    const block = input[0]; // mono Float32Array, ~128 samples, at this.inputRate

    const combined = new Float32Array(this._pending.length + block.length);
    combined.set(this._pending, 0);
    combined.set(block, this._pending.length);

    if (this.inputRate === this.targetRate) {
      this._emit(combined);
      this._pending = new Float32Array(0);
      return true;
    }

    // Linear-interpolation resample: walk the OUTPUT timeline in target-rate steps;
    // each step reads a fractional position in `combined` at the input rate.
    const outLength = Math.max(0, Math.floor((combined.length - 1) / this.ratio));
    const resampled = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const srcPos = i * this.ratio;
      const i0 = Math.floor(srcPos);
      const frac = srcPos - i0;
      resampled[i] = combined[i0] * (1 - frac) + combined[i0 + 1] * frac;
    }
    // Whatever `combined` the last output sample didn't fully consume carries into the
    // next process() call, so resampling stays continuous across 128-sample blocks
    // instead of restarting (and losing phase) every time.
    const consumedUpTo = outLength > 0 ? Math.floor((outLength - 1) * this.ratio) : 0;
    this._pending = combined.slice(consumedUpTo);
    this._emit(resampled);
    return true;
  }

  _emit(samples) {
    for (let i = 0; i < samples.length; i++) {
      this.outBuffer.push(samples[i]);
    }
    while (this.outBuffer.length >= this.chunkSamples) {
      const chunk = this.outBuffer.splice(0, this.chunkSamples);
      const int16 = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        int16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      this.port.postMessage(int16, [int16.buffer]);
    }
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
