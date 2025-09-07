document.addEventListener('DOMContentLoaded', function() {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const analyzeOutput = document.getElementById('analyzeOutput');

  // Shared rendering helpers
  const order = ['init','process_video','diarize','find_frames','summary','compose'];
  const labels = {
    init: 'Initiating gHost Mode',
    process_video: 'Processing video',
    diarize: 'Diarizing',
    find_frames: 'Finding the right frames to gHost',
    summary: 'Summary',
    compose: 'Composing final video'
  };
  let status = {};
  function render() {
    if (!analyzeOutput) return;
    const lines = [];
    let lastIdx = -1;
    for (let i = 0; i < order.length; i++) {
      const k = order[i];
      if (status[k]) lastIdx = i;
    }
    const visibleUntil = Math.min(order.length - 1, lastIdx + 1);
    for (let i = 0; i <= visibleUntil; i++) {
      const k = order[i];
      const s = status[k];
      if (s === 'completed') lines.push('✅ ' + labels[k]);
      else if (s === 'started') lines.push('⏳ ' + labels[k]);
      else lines.push('• ' + labels[k]);
    }
    analyzeOutput.textContent = lines.join('\n');
  }

  async function restoreState() {
    const { analyze_state } = await chrome.storage.local.get('analyze_state');
    if (analyze_state) {
      status = analyze_state.status || {};
      analyzeOutput.style.display = 'block';
      render();
      if (analyze_state.error) {
        analyzeOutput.textContent = 'Error: ' + analyze_state.error;
      } else if (analyze_state.final_url || analyze_state.final_path) {
        analyzeOutput.textContent += '\n\nProcessed — now gHost it';
      }
    }
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (!analyzeOutput) return;
    if (msg.type === 'stage') {
      analyzeOutput.style.display = 'block';
      status[msg.phase] = msg.status;
      render();
    } else if (msg.type === 'result') {
      // Mark summary complete but do not print JSON or text
      status['summary'] = 'completed';
      render();
    } else if (msg.type === 'error') {
      analyzeOutput.textContent = 'Error: ' + msg.error;
    } else if (msg.type === 'final') {
      // End-of-process message only; no JSON or extra text
      analyzeOutput.textContent += `\n\nProcessed — now gHost it`;
    }
  });

  // Restore any ongoing or finished state when popup opens
  restoreState();

  analyzeBtn.addEventListener('click', function() {
    analyzeOutput.style.display = 'block';
    analyzeOutput.textContent = 'Starting gHosting...';
    chrome.tabs.query({active: true, currentWindow: true}, async function(tabs) {
      const tab = tabs[0];
      if (!tab.url || !tab.url.startsWith('https://www.youtube.com/watch')) {
        alert('Please open a YouTube watch page first.');
        return;
      }
      // Open localhost dashboard when initiating gHosting
      try { await chrome.runtime.sendMessage({ action: 'openLocalhost' }); } catch (_) {}
      status = {};
      render();
      try {
        await chrome.runtime.sendMessage({ action: 'startAnalyze', url: tab.url });
      } catch (e) {
        analyzeOutput.textContent = 'Error: ' + (e && e.message ? e.message : String(e));
      }
    });
  });
});
