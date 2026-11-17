const STAGES = [
  ["endpoint_ms", "端点"],
  ["asr_ms", "ASR"],
  ["llm_first_ms", "LLM"],
  ["tts_first_ms", "TTS"],
];

export class MetricsPanel {
  constructor(ids) {
    this.ids = ids;
    this.turns = [];
    this.latest = null;
  }

  _pct(values, p) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
    return sorted[idx];
  }

  addTurn(msg) {
    this.turns.push(msg);
    this.latest = msg;
    this.renderLatest();
    this.renderTable();
    this.renderStats();
  }

  setPlayback(msg) {
    if (!this.latest || msg.turn_id !== this.latest.turn_id) return;
    this.latest.playback_ms = msg.playback_ms;
    this.latest.audible_total_ms = msg.audible_total_ms;
    this.renderLatest();
    this.renderTable();
  }

  renderLatest() {
    const msg = this.latest;
    if (!msg) return;
    const el = document.getElementById(this.ids.latest);
    el.classList.remove("latency-empty");
    el.innerHTML = `<div class="latency-total">${msg.total_ms} ms</div>
      <div class="latency-sub">服务端说完→首帧音频${msg.playback_ms != null ? ` · 客户端听感 ${msg.audible_total_ms} ms` : ""}</div>`;
    const bars = document.getElementById(this.ids.bars);
    const max = Math.max(1, ...STAGES.map(([k]) => msg[k] || 0), msg.total_ms || 0);
    bars.innerHTML = STAGES.map(([key, label]) => {
      const value = msg[key] || 0;
      const width = Math.max(2, (value / max) * 100);
      return `<div class="bar-row"><span>${label}</span><span class="bar"><i style="width:${width}%"></i></span><span>${value}ms</span></div>`;
    }).join("");
  }

  renderTable() {
    const body = document.getElementById(this.ids.rows);
    const rows = this.turns.slice(-8).reverse();
    body.innerHTML = rows.map((t, i) => {
      const n = this.turns.length - i;
      return `<tr><td>${n}</td><td>${t.endpoint_ms}</td><td>${t.asr_ms}</td><td>${t.llm_first_ms}</td><td>${t.tts_first_ms}</td><td>${t.total_ms}${t.playback_ms != null ? ` (+${t.playback_ms} 播放)` : ""}</td></tr>`;
    }).join("");
  }

  renderStats() {
    const totals = this.turns.map((t) => t.total_ms).filter((v) => v > 0);
    document.getElementById(this.ids.count).textContent = String(this.turns.length);
    document.getElementById(this.ids.p50).textContent = totals.length ? `${this._pct(totals, 50)} ms` : "-";
    document.getElementById(this.ids.p95).textContent = totals.length ? `${this._pct(totals, 95)} ms` : "-";
  }

  setServerSummary(summary) {
    const active = summary?.quota?.active_total;
    document.getElementById(this.ids.active).textContent = active == null ? "-" : String(active);
  }
}
