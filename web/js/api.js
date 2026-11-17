const jsonHeaders = (key) => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${key}`,
});

export async function getTicket(key) {
  const resp = await fetch("/v1/realtime/ticket", { method: "POST", headers: jsonHeaders(key) });
  if (!resp.ok) throw new Error(`ticket: HTTP ${resp.status}`);
  return resp.json();
}

export async function getVoices() {
  try {
    const resp = await fetch("/v1/voices");
    if (!resp.ok) return [];
    return (await resp.json()).voices || [];
  } catch {
    return [];
  }
}

export async function getSummary() {
  try {
    const resp = await fetch("/v1/metrics/summary");
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}
