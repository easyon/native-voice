// Schedule server PCM16 (24 kHz) chunks with WebAudio.
export class Player {
  constructor() {
    this.ctx = new AudioContext();
    this.sources = new Set();
    this.nextTime = 0;
    this.backchannelNextTime = 0;
    this.turnId = null;
    this.startedSent = false;
    this.allSent = false;
    this.onStarted = null;
    this.onDone = null;
    this._timer = null;
    // 所有播放音频都经过 analyser，用于驱动通话界面的呼吸圈。
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.7;
    this.analyser.connect(this.ctx.destination);
    this._levelBuf = new Float32Array(this.analyser.fftSize);
  }

  async resume() {
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  // 当前实际播出的音频能量（RMS）。
  level() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this._levelBuf);
    const buf = this._levelBuf;
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
  }

  beginTurn(turnId, onStarted, onDone) {
    this.stopAll(false);
    this.turnId = turnId;
    this.startedSent = false;
    this.allSent = false;
    this.onStarted = onStarted;
    this.onDone = onDone;
    this.nextTime = this.ctx.currentTime + 0.02;
    if (this._timer) clearInterval(this._timer);
    this._timer = setInterval(() => this._checkDone(), 100);
  }

  push(arrayBuffer) {
    if (!this.turnId) return;
    const pcm = new Int16Array(arrayBuffer);
    const data = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
    const buffer = this.ctx.createBuffer(1, data.length, 24000);
    buffer.copyToChannel(data, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.analyser);
    const at = Math.max(this.ctx.currentTime + 0.01, this.nextTime);
    source.start(at);
    this.nextTime = at + buffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
    if (!this.startedSent) {
      this.startedSent = true;
      const delay = Math.max(0, (at - this.ctx.currentTime) * 1000);
      setTimeout(() => this.onStarted?.(), delay);
    }
  }

  // Short acknowledgement played at low gain while the user is still speaking.
  // It is deliberately outside the turn queue and never reports playback.
  pushBackchannel(arrayBuffer, volume = 0.45) {
    if (!arrayBuffer || arrayBuffer.byteLength === 0) return;
    const pcm = new Int16Array(arrayBuffer);
    const data = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
    const buffer = this.ctx.createBuffer(1, data.length, 24000);
    buffer.copyToChannel(data, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    const gain = this.ctx.createGain();
    gain.gain.value = Math.max(0.1, Math.min(1, Number(volume) || 0.45));
    source.connect(gain).connect(this.analyser);
    const at = Math.max(this.ctx.currentTime + 0.01, this.backchannelNextTime);
    source.start(at);
    this.backchannelNextTime = at + buffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  markAllSent() {
    this.allSent = true;
    this._checkDone();
  }

  _checkDone() {
    if (!this.allSent) return;
    if (this.ctx.currentTime >= this.nextTime - 0.02 && this.sources.size === 0) {
      const turnId = this.turnId;
      this._clearTimer();
      this.turnId = null;
      this.onDone?.(turnId);
    }
  }

  _clearTimer() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
  }

  stopAll(notify = true) {
    for (const source of this.sources) {
      try { source.stop(); } catch {}
    }
    this.sources.clear();
    this.backchannelNextTime = 0;
    this._clearTimer();
    this.turnId = null;
    this.allSent = false;
    this.startedSent = false;
  }
}
