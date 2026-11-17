// 通话呼吸圈：粒子球体 + 状态配色 + 随麦克风 / AI 播放音频振幅起伏。
const PALETTES = {
  idle: { light: [169, 196, 216], dim: [77, 106, 131] },
  listening: { light: [154, 230, 255], dim: [44, 140, 194] },
  thinking: { light: [255, 224, 160], dim: [184, 120, 28] },
  speaking: { light: [166, 255, 228], dim: [31, 158, 118] },
};

export class VoiceOrb {
  constructor(el, sources = {}) {
    this.el = el;
    this.canvas = el.querySelector(".orb-core");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.sources = sources;
    this.state = "idle";
    this.level = 0;
    this.width = 0;
    this.height = 0;
    this.particles = [];
    for (let i = 0; i < 260; i++) {
      const u = Math.random() * 2 - 1;
      const phi = Math.random() * Math.PI * 2;
      const r = Math.sqrt(1 - u * u);
      this.particles.push({
        x: r * Math.cos(phi),
        y: u,
        z: r * Math.sin(phi),
        seed: Math.random() * Math.PI * 2,
      });
    }
    this._resize();
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(() => this._resize());
      this._ro.observe(el);
    } else {
      window.addEventListener("resize", () => this._resize());
    }
    this._tick = this._tick.bind(this);
    this.el.style.setProperty("--level", "0");
    this._raf = requestAnimationFrame(this._tick);
  }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.el.className = `orb-stage ${state}`;
  }

  _resize() {
    if (!this.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = this.el.clientWidth || 160;
    const h = this.el.clientHeight || 160;
    if (!w || !h) return;
    this.width = w;
    this.height = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // 音频 RMS 约 0.006（静音）~ 0.25（正常说话），映射到 0~1 的视觉强度。
  _normalize(rms) {
    const v = (rms - 0.008) / 0.16;
    return Math.min(1, Math.max(0, v));
  }

  _target(now) {
    switch (this.state) {
      case "listening":
        return this._normalize(this.sources.mic?.() || 0);
      case "speaking":
        return Math.min(1, this._normalize(this.sources.ai?.() || 0) * 1.25);
      case "thinking":
        return 0.2 + 0.1 * Math.sin(now / 260);
      default:
        return 0.05 + 0.045 * Math.sin(now / 1100);
    }
  }

  _tick(now) {
    const target = Math.max(0, Math.min(1, this._target(now)));
    // 起音快、回落慢，接近人声的呼吸包络。
    const k = target > this.level ? 0.45 : 0.08;
    this.level += (target - this.level) * k;
    this.el.style.setProperty("--level", this.level.toFixed(3));
    this._draw(now);
    this._raf = requestAnimationFrame(this._tick);
  }

  _draw(now) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    if (!ctx || !w || !h) return;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.4;
    const pal = PALETTES[this.state] || PALETTES.idle;
    const t = now / 1000;
    const amp = this.level;

    // 球心柔光（仅作为光晕，不使用实心球体）
    const glow = ctx.createRadialGradient(cx, cy, radius * 0.1, cx, cy, radius * 1.3);
    glow.addColorStop(0, `rgba(${pal.light[0]}, ${pal.light[1]}, ${pal.light[2]}, ${0.08 + amp * 0.18})`);
    glow.addColorStop(0.6, `rgba(${pal.dim[0]}, ${pal.dim[1]}, ${pal.dim[2]}, ${0.05 + amp * 0.08})`);
    glow.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 1.3, 0, Math.PI * 2);
    ctx.fill();

    const yaw = t * 0.32;
    const pitch = Math.sin(t * 0.23) * 0.22;
    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);
    const cosX = Math.cos(pitch);
    const sinX = Math.sin(pitch);
    const spread = 1 + amp * 0.16;

    for (const p of this.particles) {
      let x = p.x * cosY - p.z * sinY;
      let z = p.x * sinY + p.z * cosY;
      const y = p.y * cosX - z * sinX;
      z = p.y * sinX + z * cosX;

      const depth = (z + 1) / 2; // 0 = 远端, 1 = 近端
      const jitter = 1 + 0.014 * Math.sin(t * 1.9 + p.seed);
      const px = cx + x * radius * spread * jitter;
      const py = cy + y * radius * spread * jitter;
      const alpha = Math.max(
        0.04,
        Math.min(0.95, (0.1 + depth * 0.62) * (0.62 + amp * 0.5) + 0.05 * Math.sin(t * 2.4 + p.seed))
      );
      const size = (0.8 + depth * 1.8) * (1 + amp * 0.75);
      const r = Math.round(pal.dim[0] + (pal.light[0] - pal.dim[0]) * depth);
      const g = Math.round(pal.dim[1] + (pal.light[1] - pal.dim[1]) * depth);
      const b = Math.round(pal.dim[2] + (pal.light[2] - pal.dim[2]) * depth);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.beginPath();
      ctx.arc(px, py, size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    this._raf = null;
    if (this._ro) this._ro.disconnect();
  }
}
