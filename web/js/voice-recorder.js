// Microphone recorder for cloning. Opens its own stream with browser DSP off so
// the reference is as unprocessed as possible. Returns Float32 chunks + rate.
export class VoiceRecorder {
  constructor() {
    this.stream = null;
    this.ctx = null;
    this.node = null;
    this.source = null;
    this.sink = null;
    this.chunks = [];
    this.recording = false;
  }

  async start() {
    if (this.recording) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
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
    this.sink.gain.value = 0;
    this.source.connect(this.node);
    this.node.connect(this.sink);
    this.sink.connect(this.ctx.destination);
    this.chunks = [];
    this.node.port.onmessage = (event) => {
      if (this.recording) this.chunks.push(event.data.slice(0));
    };
    this.recording = true;
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  async stop() {
    if (!this.recording) return { chunks: [], sampleRate: 16000, seconds: 0 };
    this.recording = false;
    const sampleRate = this.ctx?.sampleRate || 16000;
    const chunks = this.chunks;
    this.chunks = [];
    try { this.node?.disconnect(); } catch {}
    try { this.sink?.disconnect(); } catch {}
    try { this.source?.disconnect(); } catch {}
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    if (this.ctx) await this.ctx.close();
    this.stream = this.ctx = this.node = this.source = this.sink = null;
    const samples = chunks.reduce((n, c) => n + c.length, 0);
    return { chunks, sampleRate, seconds: samples / sampleRate };
  }
}
