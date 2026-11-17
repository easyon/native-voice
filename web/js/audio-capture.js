// Microphone -> 16 kHz mono PCM16 frames (20 ms).
export class MicCapture {
  constructor() {
    this.stream = null;
    this.ctx = null;
    this.node = null;
    this.source = null;
    this.sink = null;
    this.buffer = new Float32Array(0);
    this.onFrame = null;
    this.active = false;
  }

  async start(onFrame) {
    this.onFrame = onFrame;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    try {
      this.ctx = new AudioContext({ sampleRate: 16000 });
    } catch {
      this.ctx = new AudioContext();
    }
    await this.ctx.audioWorklet.addModule("/ui/pcm-worklet.js");
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-processor");
    this.sink = this.ctx.createGain();
    this.sink.gain.value = 0; // keep the graph running without monitoring
    this.source.connect(this.node);
    this.node.connect(this.sink);
    this.sink.connect(this.ctx.destination);
    this.node.port.onmessage = (event) => this._push(event.data);
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.active = true;
  }

  _push(samples) {
    let data = samples;
    if (this.ctx.sampleRate !== 16000) {
      const ratio = this.ctx.sampleRate / 16000;
      const out = new Float32Array(Math.floor(data.length / ratio));
      for (let i = 0; i < out.length; i++) out[i] = data[Math.floor(i * ratio)] || 0;
      data = out;
    }
    const merged = new Float32Array(this.buffer.length + data.length);
    merged.set(this.buffer);
    merged.set(data, this.buffer.length);
    this.buffer = merged;
    const frame = 320; // 20 ms @ 16 kHz
    while (this.buffer.length >= frame) {
      const chunk = this.buffer.slice(0, frame);
      this.buffer = this.buffer.slice(frame);
      this.onFrame?.(chunk);
    }
  }

  async stop() {
    this.active = false;
    try { this.node?.disconnect(); } catch {}
    try { this.sink?.disconnect(); } catch {}
    try { this.source?.disconnect(); } catch {}
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    if (this.ctx) await this.ctx.close();
    this.stream = this.ctx = this.node = this.source = this.sink = null;
    this.buffer = new Float32Array(0);
  }
}

export function floatToPcm16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return out.buffer;
}
