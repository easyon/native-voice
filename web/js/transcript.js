export class Transcript {
  constructor(root, partialEl) {
    this.root = root;
    this.partialEl = partialEl;
    this.currentAi = null;
  }

  _scroll() {
    this.root.scrollTop = this.root.scrollHeight;
  }

  _node(cls, text = "") {
    const el = document.createElement("div");
    el.className = `msg ${cls}`;
    el.textContent = text;
    return el;
  }

  setPartial(text) {
    this.partialEl.hidden = !text;
    this.partialEl.textContent = text ? `识别中：${text}` : "";
  }

  addUser(text, meta = "") {
    this.setPartial("");
    const el = this._node("user", text);
    if (meta) {
      const m = document.createElement("span");
      m.className = "meta";
      m.textContent = meta;
      el.appendChild(m);
    }
    this.root.appendChild(el);
    this._scroll();
  }

  appendAssistant(delta) {
    if (!this.currentAi) {
      this.currentAi = this._node("ai", "");
      this.root.appendChild(this.currentAi);
    }
    this.currentAi.textContent += delta;
    this._scroll();
  }

  finishAssistant() {
    this.currentAi = null;
  }

  note(text) {
    this.root.appendChild(this._node("note", text));
    this._scroll();
  }

  clear() {
    this.root.innerHTML = "";
    this.currentAi = null;
    this.setPartial("");
  }
}
