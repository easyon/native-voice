import { getSummary, getTicket, getVoices, wsUrl } from "./api.js";
import { MicCapture, floatToPcm16 } from "./audio-capture.js";
import { Player } from "./audio-playback.js";
import { Transcript } from "./transcript.js";
import { MetricsPanel } from "./metrics.js";
import { Waveform } from "./waveform.js";
import { VoiceOrb } from "./voice-orb.js";
import { VoiceRecorder } from "./voice-recorder.js";
import { encodeWav } from "./wav.js";

const $ = (id) => document.getElementById(id);
const THEME_KEY = "nv_theme";

function applyTheme(theme) {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem(THEME_KEY, next);
  const btn = $("themeBtn");
  if (btn) {
    const label = next === "light" ? "切换到暗色主题" : "切换到亮色主题";
    btn.setAttribute("aria-label", label);
    btn.title = label;
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "light" ? "dark" : "light");
}

const transcript = new Transcript($("transcript"), $("partial"));
const metrics = new MetricsPanel({
  latest: "latestLatency",
  bars: "latencyBars",
  rows: "turnRows",
  count: "turnsCount",
  p50: "p50",
  p95: "p95",
  active: "activeSessions",
});
const mic = new MicCapture();
const player = new Player();
const waveform = new Waveform($("audioWave"));
const orb = new VoiceOrb($("orbStage"), {
  mic: () => waveform.micLevel,
  ai: () => player.level(),
});
const recorder = new VoiceRecorder();

let ws = null;
let micOn = false;
let recordedBlob = null;
let recording = false;
let recordTimer = null;
let recordStartedAt = 0;
let currentTurnId = null;
let backchannelActive = false;
let backchannelVolume = 0.45;
let summaryTimer = null;
let errorTimer = null;
let callStartedAt = 0;
let callInterval = null;
let hangingUp = false;
let aiBuffer = "";

function formatClock(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function updateCallTimer() {
  $("callTimer").textContent = formatClock(Date.now() - callStartedAt);
}

function startCallTimer() {
  stopCallTimer();
  callStartedAt = Date.now();
  updateCallTimer();
  callInterval = setInterval(updateCallTimer, 1000);
}

function stopCallTimer() {
  if (callInterval) clearInterval(callInterval);
  callInterval = null;
  callStartedAt = 0;
  $("callTimer").textContent = "00:00";
}

function orbStateFor(text, kind) {
  if (kind === "ready") return "listening";
  if (text.includes("说话")) return "speaking";
  if (text.includes("断开") || text.includes("挂断") || text.includes("未连接") || text.includes("关闭")) return "idle";
  if (text.includes("思考") || text.includes("连接") || text.includes("获取") || text.includes("呼叫")) return "thinking";
  return "idle";
}

function setStatus(text, kind = "") {
  $("statusText").textContent = text;
  const dot = $("statusDot");
  dot.className = "dot" + (kind ? ` ${kind}` : "");
  const state = $("callState");
  if (state) state.textContent = text;
  orb.setState(orbStateFor(text, kind));
}

function _renderCaption(el, who, text, partial) {
  if (!el) return;
  el.textContent = "";
  el.classList.remove("show", "partial");
  if (!text) return;
  const trimmed = text.length > 140 ? `…${text.slice(-140)}` : text;
  const tag = document.createElement("b");
  tag.className = who === "user" ? "cap-user" : "cap-ai";
  tag.textContent = who === "user" ? "你" : "AI";
  el.append(tag, " ", trimmed);
  el.classList.toggle("partial", Boolean(partial));
  el.classList.add("show");
}

// 用户字幕常驻显示：识别中用斜体弱化，final 后保留到下一次开口。
function setUserCaption(text, partial = false) {
  _renderCaption($("liveUser"), "user", text, partial);
}

function setAiCaption(text) {
  _renderCaption($("liveCaption"), "ai", text);
}

function clearCaptions() {
  setUserCaption("");
  setAiCaption("");
}

function showError(message) {
  const el = $("error");
  el.hidden = false;
  el.textContent = message;
  if (errorTimer) clearTimeout(errorTimer);
  errorTimer = setTimeout(() => { el.hidden = true; }, 6000);
}

function setCallConnecting(on) {
  const btn = $("callBtn");
  btn.classList.toggle("connecting", on);
  btn.disabled = on;
  if (on) {
    btn.setAttribute("aria-label", "正在呼叫");
    $("callLabel").textContent = "呼叫中…";
  }
}

function setConnected(on) {
  const btn = $("callBtn");
  btn.classList.remove("connecting");
  btn.classList.toggle("active", on);
  btn.disabled = false;
  btn.setAttribute("aria-label", on ? "挂断" : "呼叫");
  $("callLabel").textContent = on ? "挂断" : "呼叫";
  $("micBtn").disabled = !on;
  if (!on) {
    stopCallTimer();
    setStatus("未连接");
    waveform.reset();
    clearCaptions();
  }
}

function updateMicButton() {
  const btn = $("micBtn");
  btn.classList.toggle("muted", !micOn);
  btn.title = micOn ? "暂停麦克风" : "开启麦克风";
  btn.setAttribute("aria-label", btn.title);
  btn.setAttribute("aria-pressed", String(micOn));
}

function updatePromptCount() {
  $("promptCount").textContent = `${$("systemPrompt").value.length} / 2000`;
}

function applyPrompt() {
  const text = $("systemPrompt").value;
  localStorage.setItem("nv_system_prompt", text);
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "session.config", system_prompt: text }));
    transcript.note("角色提示词已更新，从下一轮开始生效");
  }
}

async function refreshVoices() {
  const voices = await getVoices();
  const select = $("voice");
  const current = select.value || "default";
  select.innerHTML = "";
  for (const v of voices.length ? voices : [{ name: "default", description: "default" }]) {
    const opt = document.createElement("option");
    opt.value = v.name;
    opt.textContent = v.description ? `${v.name} · ${v.description}` : v.name;
    select.appendChild(opt);
  }
  select.value = voices.some((v) => v.name === current) ? current : (voices[0]?.name || "default");
}

async function connect() {
  const key = ($("apiKey").value || sessionStorage.getItem("nv_key") || "").trim();
  if (!key) {
    showError("请先在左侧填写 API Key");
    $("apiKey").focus();
    return;
  }
  sessionStorage.setItem("nv_key", key);
  localStorage.setItem("nv_system_prompt", $("systemPrompt").value);
  setCallConnecting(true);
  setStatus("获取连接票据…", "busy");
  try {
    const [ticket] = await Promise.all([getTicket(key), refreshVoices()]);
    await player.resume();
    hangingUp = false;
    const socket = new WebSocket(wsUrl("/ws/conversation"));
    ws = socket;
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "auth", ticket: ticket.ticket }));
    };
    socket.onmessage = (event) => handleMessage(event);
    socket.onerror = () => showError("WebSocket 连接失败");
    socket.onclose = () => {
      if (ws !== socket) return;
      ws = null;
      currentTurnId = null;
      backchannelActive = false;
      aiBuffer = "";
      setConnected(false);
      stopMic();
      stopCallTimer();
      clearCaptions();
      if (summaryTimer) clearInterval(summaryTimer);
      summaryTimer = null;
      setStatus(hangingUp ? "已挂断" : "连接已关闭");
      hangingUp = false;
    };
  } catch (err) {
    setCallConnecting(false);
    setConnected(false);
    showError(String(err.message || err));
    setStatus("连接失败");
  }
}

function handleMessage(event) {
  if (typeof event.data !== "string") {
    if (backchannelActive) {
      player.pushBackchannel(event.data, backchannelVolume);
      return;
    }
    if (currentTurnId) {
      player.push(event.data);
    }
    return;
  }
  let msg;
  try { msg = JSON.parse(event.data); } catch { return; }
  switch (msg.type) {
    case "session.ready":
      setConnected(true);
      startCallTimer();
      aiBuffer = "";
      setStatus("已连接，正在听", "ready");
      clearCaptions();
      if (!$("systemPrompt").value && msg.system_prompt) {
        $("systemPrompt").value = msg.system_prompt;
        updatePromptCount();
      }
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "session.start",
            voice: $("voice").value,
            endpoint_ms: Number($("endpointMs").value),
            system_prompt: $("systemPrompt").value,
          })
        );
      }
      summaryTimer = setInterval(async () => metrics.setServerSummary(await getSummary()), 3000);
      startMic();
      break;
    case "asr.partial":
      transcript.setPartial(msg.text || "");
      setUserCaption(msg.text || "", true);
      break;
    case "asr.final": {
      transcript.finishAssistant();
      aiBuffer = "";
      const duration = Math.max(0, msg.end_ms - msg.begin_ms) / 1000;
      transcript.addUser(
        msg.text,
        `句 ${msg.sentence_id} · 时长 ${duration.toFixed(2)}s · 会话 ${formatClock(msg.begin_ms)}–${formatClock(msg.end_ms)}`
      );
      setUserCaption(msg.text);
      setStatus("思考中…", "busy");
      break;
    }
    case "assistant.text.delta":
      transcript.appendAssistant(msg.text || "");
      aiBuffer += msg.text || "";
      setAiCaption(aiBuffer);
      break;
    case "assistant.text.done":
      transcript.finishAssistant();
      setAiCaption(aiBuffer);
      break;
    case "assistant.tool.call":
      transcript.note(`正在调用工具：${msg.name}`);
      setStatus("调用工具中…", "busy");
      break;
    case "assistant.tool.result": {
      const result = String(msg.result || "").replace(/\s+/g, " ").slice(0, 80);
      transcript.note(msg.ok ? `工具结果：${result}` : `工具失败：${result}`);
      setStatus("思考中…", "busy");
      break;
    }
    case "tts.audio.start":
      backchannelActive = false;
      if (msg.turn_id !== currentTurnId) {
        currentTurnId = msg.turn_id;
        player.beginTurn(
          msg.turn_id,
          () => {
            ws?.send(JSON.stringify({ type: "playback.started", turn_id: msg.turn_id }));
            setStatus("AI 说话中（可直接插话）", "busy");
          },
          (turnId) => {
            ws?.send(JSON.stringify({ type: "playback.done", turn_id: turnId }));
            setStatus("聆听中", "ready");
          }
        );
      }
      break;
    case "assistant.audio.all_done":
      player.markAllSent();
      break;
    case "assistant.backchannel.start":
      backchannelActive = true;
      backchannelVolume = Number.isFinite(msg.volume) ? msg.volume : 0.45;
      break;
    case "assistant.backchannel.done":
      backchannelActive = false;
      break;
    case "assistant.interrupted":
      backchannelActive = false;
      player.stopAll();
      transcript.finishAssistant();
      transcript.note("已打断，继续听你说…");
      setAiCaption("已打断，继续听你说…");
      setStatus("聆听中", "ready");
      break;
    case "assistant.cancelled":
      player.stopAll();
      transcript.finishAssistant();
      transcript.note("上一段语音已合并，继续识别…");
      break;
    case "metrics.turn":
      metrics.addTurn(msg);
      break;
    case "metrics.playback":
      metrics.setPlayback(msg);
      break;
    case "error":
      showError(msg.message || msg.code || "服务端错误");
      break;
    default:
      break;
  }
}

async function startMic() {
  if (micOn || !ws) return;
  try {
    await mic.start((frame) => {
      waveform.pushMic(frame);
      if (ws?.readyState === WebSocket.OPEN && micOn) {
        ws.send(floatToPcm16(frame));
      }
    });
    micOn = true;
    updateMicButton();
    setStatus("聆听中", "ready");
  } catch (err) {
    micOn = false;
    updateMicButton();
    showError(`麦克风不可用：${err.message || err}`);
  }
}

async function stopMic() {
  if (!micOn) return;
  micOn = false;
  updateMicButton();
  await mic.stop();
}

async function toggleMic() {
  if (micOn) {
    await stopMic();
    setStatus("麦克风已暂停");
  } else {
    await startMic();
  }
}

async function toggleRecord() {
  if (recording) {
    await stopRecording();
    return;
  }
  try {
    await recorder.start();
    recording = true;
    recordStartedAt = Date.now();
    $("recordBtn").textContent = "停止录音";
    $("recordStatus").textContent = "录音中 00:00（建议 3–10 秒）";
    if (recordTimer) clearInterval(recordTimer);
    recordTimer = setInterval(() => {
      const ms = Date.now() - recordStartedAt;
      $("recordStatus").textContent = `录音中 ${formatClock(ms)}（建议 3–10 秒）`;
      if (ms >= 30000) stopRecording();
    }, 250);
  } catch (err) {
    $("recordStatus").textContent = `无法录音：${err.message || err}`;
  }
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  if (recordTimer) clearInterval(recordTimer);
  recordTimer = null;
  $("recordBtn").textContent = "开始录音";
  try {
    const { chunks, sampleRate, seconds } = await recorder.stop();
    if (seconds < 1.5) {
      recordedBlob = null;
      $("recordStatus").textContent = `录音 ${seconds.toFixed(1)} 秒太短，请至少 1.5 秒`;
      return;
    }
    recordedBlob = encodeWav(chunks, sampleRate);
    $("cloneFile").value = "";
    $("recordStatus").textContent = `已录音 ${seconds.toFixed(1)} 秒 · 可上传克隆`;
  } catch (err) {
    $("recordStatus").textContent = `录音失败：${err.message || err}`;
  }
}

function clearRecording() {
  if (recording) stopRecording();
  recordedBlob = null;
  $("cloneFile").value = "";
  $("recordStatus").textContent = "未录音 · 建议 3–10 秒，请在安静环境录制，避免录到 AI 声音";
}

function openCloneDialog() {
  $("cloneStatus").textContent = "";
  const dialog = $("cloneDialog");
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
}

function closeCloneDialog() {
  const dialog = $("cloneDialog");
  if (dialog.open) dialog.close();
  else dialog.removeAttribute("open");
}

async function cloneVoice() {
  const key = ($("apiKey").value || sessionStorage.getItem("nv_key") || "").trim();
  const file = recordedBlob
    ? new File([recordedBlob], "recording.wav", { type: "audio/wav" })
    : $("cloneFile").files[0];
  if (!key) {
    showError("请先填写 API Key");
    return;
  }
  if (!file) {
    $("cloneStatus").textContent = "请先选择参考音频或录制一段";
    return;
  }
  if (!$("cloneConsent").checked) {
    $("cloneStatus").textContent = "请勾选声音授权确认";
    return;
  }
  const form = new FormData();
  form.append("file", file, file.name || "reference.wav");
  form.append("name", $("cloneName").value || "我的音色");
  form.append("text", $("cloneText").value || "");
  form.append("consent", "true");
  $("cloneBtn").disabled = true;
  $("cloneStatus").textContent = "上传并克隆中…";
  try {
    const resp = await fetch("/v1/voices/clone", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}` },
      body: form,
    });
    const obj = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(obj.detail || `HTTP ${resp.status}`);
    await refreshVoices();
    $("voice").value = obj.voice.name;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "session.config", voice: obj.voice.name }));
    }
    const preview = obj.transcript.length > 30 ? `${obj.transcript.slice(0, 30)}…` : obj.transcript;
    $("cloneStatus").textContent = `克隆完成：${obj.voice.display_name} · ${obj.duration}s · 参考文本：${preview}`;
    recordedBlob = null;
    $("cloneFile").value = "";
    $("recordStatus").textContent = "未录音 · 建议 3–10 秒，请在安静环境录制，避免录到 AI 声音";
    transcript.note(`音色已克隆：${obj.voice.display_name}，下一句回答起使用`);
    closeCloneDialog();
  } catch (err) {
    $("cloneStatus").textContent = `克隆失败：${err.message || err}`;
  } finally {
    $("cloneBtn").disabled = false;
  }
}

async function deleteSelectedVoice() {
  const key = ($("apiKey").value || sessionStorage.getItem("nv_key") || "").trim();
  const voice = $("voice").value;
  if (!key) {
    showError("请先填写 API Key");
    return;
  }
  if (!voice.startsWith("clone_")) {
    $("cloneStatus").textContent = "当前音色不是克隆音色";
    return;
  }
  if (!confirm(`确定删除音色 ${voice}？`)) return;
  $("deleteVoiceBtn").disabled = true;
  try {
    const resp = await fetch(`/v1/voices/${encodeURIComponent(voice)}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${key}` },
    });
    const obj = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(obj.detail || `HTTP ${resp.status}`);
    await refreshVoices();
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "session.config", voice: $("voice").value }));
    }
    $("cloneStatus").textContent = `已删除 ${voice}`;
  } catch (err) {
    $("cloneStatus").textContent = `删除失败：${err.message || err}`;
  } finally {
    $("deleteVoiceBtn").disabled = false;
  }
}

function toggleCall() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    disconnect();
    return;
  }
  connect();
}

function disconnect() {
  setStatus("正在挂断…", "busy");
  hangingUp = Boolean(ws);
  try { ws?.send(JSON.stringify({ type: "session.end" })); } catch {}
  try { ws?.close(); } catch {}
  player.stopAll();
  stopMic();
  stopCallTimer();
  aiBuffer = "";
  clearCaptions();
  setConnected(false);
  setStatus("已挂断");
  if (!ws) hangingUp = false;
}

async function boot() {
  applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
  $("themeBtn").addEventListener("click", toggleTheme);
  const saved = sessionStorage.getItem("nv_key");
  if (saved) $("apiKey").value = saved;
  const savedPrompt = localStorage.getItem("nv_system_prompt");
  if (savedPrompt) $("systemPrompt").value = savedPrompt;
  updatePromptCount();
  updateMicButton();
  $("systemPrompt").addEventListener("input", updatePromptCount);
  $("applyPrompt").addEventListener("click", applyPrompt);
  $("cloneBtn").addEventListener("click", cloneVoice);
  $("deleteVoiceBtn").addEventListener("click", deleteSelectedVoice);
  $("recordBtn").addEventListener("click", toggleRecord);
  $("clearRecordBtn").addEventListener("click", clearRecording);
  $("cloneOpenBtn").addEventListener("click", openCloneDialog);
  $("cloneCloseBtn").addEventListener("click", closeCloneDialog);
  $("cloneDialog").addEventListener("click", (event) => {
    if (event.target === $("cloneDialog")) closeCloneDialog();
  });
  $("cloneDialog").addEventListener("close", () => {
    if (recording) stopRecording();
  });
  $("cloneFile").addEventListener("change", () => {
    if (recording) stopRecording();
    recordedBlob = null;
    $("recordStatus").textContent = $("cloneFile").files[0]
      ? "已选择上传文件"
      : "未录音 · 建议 3–10 秒，请在安静环境录制，避免录到 AI 声音";
  });
  requestAnimationFrame(() => waveform._resize());
  await refreshVoices();
  $("callBtn").addEventListener("click", toggleCall);
  $("micBtn").addEventListener("click", toggleMic);
  $("clearBtn").addEventListener("click", () => {
    transcript.clear();
    clearCaptions();
  });
  $("logToggle").addEventListener("click", () => {
    const active = document.querySelector(".call-stage").classList.toggle("log-view");
    $("logToggle").setAttribute("aria-pressed", String(active));
    $("logToggle").title = active ? "返回通话视图" : "切换到转写记录";
    if (active) {
      const log = $("transcript");
      log.scrollTop = log.scrollHeight;
    }
  });
  $("endpointMs").addEventListener("change", () => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "session.config", endpoint_ms: Number($("endpointMs").value) }));
    }
  });
}

boot();
