// MV3 service worker to run streaming analyze even if popup closes

const STATE_KEY = 'analyze_state';
let current = null; // { controller, status, result, startedAt }
let localhostOpened = false; // ensure we only open once per service-worker lifetime

async function saveState(partial) {
  const prev = (await chrome.storage.local.get(STATE_KEY))[STATE_KEY] || {};
  const next = { ...prev, ...partial };
  await chrome.storage.local.set({ [STATE_KEY]: next });
  return next;
}

async function clearState() {
  await chrome.storage.local.remove(STATE_KEY);
}

function broadcast(msg) {
  try { chrome.runtime.sendMessage({ source: 'background', ...msg }); } catch (_) {}
}

async function startAnalyze({ url, prompt }) {
  // Abort existing run if any
  if (current && current.controller) {
    try { current.controller.abort(); } catch (_) {}
  }
  const controller = new AbortController();
  current = { controller, status: {}, result: null, startedAt: Date.now() };
  await saveState({ running: true, status: {}, result: null, url, prompt: prompt || null });
  broadcast({ type: 'analyze:start', url });

  try {
    const res = await fetch('http://127.0.0.1:8000/analyze_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, ...(prompt ? { prompt } : {}) }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error('Backend error: ' + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        try {
          const msg = JSON.parse(line);
          // Persist and broadcast key events
          if (msg.type === 'stage') {
            current.status[msg.phase] = msg.status;
            await saveState({ status: current.status });
            broadcast({ type: 'stage', phase: msg.phase, status: msg.status, label: msg.label });
          } else if (msg.type === 'result') {
            current.result = msg.llm || msg;
            await saveState({ status: { ...current.status, summary: 'completed' }, result: current.result });
            broadcast({ type: 'result', llm: current.result });
          } else if (msg.type === 'final') {
            const st = await saveState({ final_path: msg.final_path, final_url: msg.final_url, video_id: msg.video_id });
            broadcast({ type: 'final', final_path: st.final_path, final_url: st.final_url, video_id: st.video_id });
          } else if (msg.type === 'error') {
            await saveState({ error: msg.error, running: false });
            broadcast({ type: 'error', error: msg.error });
          }
        } catch (_) {
          // ignore parse errors
        }
      }
    }
  } catch (e) {
    const err = e && e.message ? e.message : String(e);
    await saveState({ error: err, running: false });
    broadcast({ type: 'error', error: err });
  } finally {
    await saveState({ running: false });
    current = null;
    broadcast({ type: 'analyze:done' });
  }
}

chrome.runtime.onMessage.addListener((req, _sender, sendResponse) => {
  (async () => {
    if (req && req.action === 'startAnalyze') {
      await startAnalyze({ url: req.url, prompt: req.prompt });
      sendResponse({ ok: true });
    } else if (req && req.action === 'getAnalyzeState') {
      const state = (await chrome.storage.local.get(STATE_KEY))[STATE_KEY] || {};
      sendResponse({ ok: true, state });
    } else if (req && req.action === 'cancelAnalyze') {
      if (current && current.controller) {
        try { current.controller.abort(); } catch (_) {}
      }
      await saveState({ running: false, canceled: true });
      sendResponse({ ok: true });
    } else if (req && req.action === 'openLocalhost') {
      if (!localhostOpened) {
        try {
          chrome.tabs.create({ url: 'http://localhost:7000' }, () => {
            const err = chrome.runtime.lastError;
            if (err) {
              sendResponse({ ok: false, error: err.message || String(err) });
            } else {
              localhostOpened = true;
              sendResponse({ ok: true });
            }
          });
          return true;
        } catch (e) {
          sendResponse({ ok: false, error: e && e.message ? e.message : String(e) });
        }
      } else {
        sendResponse({ ok: true, alreadyOpened: true });
      }
    }
  })();
  return true; // keep channel open for async response
});
