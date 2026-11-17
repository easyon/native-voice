// Minimal oscillating waveform for the local microphone. It is drawn in the
// centre of the call stage and only reacts to microphone input; TTS output is
// intentionally not visualised.
export class Waveform {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.micLevel = 0;
    this.level = 0;
    this.target = 0;
    this.phase = 0;
    this.width = 0;
    this.height = 0;
    this._resize();
    window.addEventListener("resize", () => this._resize());
    const loop = () => {
      this.level += (this.target - this.level) * 0.12;
      this.phase += 0.05 + this.level * 0.06;
      this._draw();
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  _resize() {
    if (!this.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(160, this.canvas.clientWidth || 320);
    const h = Math.max(24, this.canvas.clientHeight || 52);
    this.width = w;
    this.height = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  pushMic(frame) {
    if (!frame?.length) return;
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);
    this.micLevel = rms;
    this.target = Math.min(1, Math.sqrt(rms) * 2.4);
  }

  reset() {
    this.micLevel = 0;
    this.target = 0;
    this.level = 0;
    this.phase = 0;
  }

  _draw() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    const mid = h / 2;
    ctx.clearRect(0, 0, w, h);

    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "rgba(76, 194, 255, 0)");
    grad.addColorStop(0.18, "rgba(76, 194, 255, 0.85)");
    grad.addColorStop(0.5, "rgba(51, 214, 159, 0.95)");
    grad.addColorStop(0.82, "rgba(76, 194, 255, 0.85)");
    grad.addColorStop(1, "rgba(76, 194, 255, 0)");

    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.6;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let x = 0; x <= w; x += 2) {
      const env = Math.pow(Math.sin((Math.PI * x) / w), 0.7);
      const y =
        mid +
        (Math.sin(x * 0.045 + this.phase) * 0.8 +
          Math.sin(x * 0.11 - this.phase * 1.4) * 0.35) *
          this.level *
          env *
          h *
          0.42;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
