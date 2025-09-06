document.addEventListener('DOMContentLoaded', function() {
  const replaceBtn = document.getElementById('replaceBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const analyzeOutput = document.getElementById('analyzeOutput');
  const videoFileInput = document.getElementById('videoFile');
  const fileLabel = document.getElementById('fileLabel');

  // Update label when file is selected
  videoFileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
      fileLabel.textContent = this.files[0].name;
    } else {
      fileLabel.textContent = 'No video chosen';
    }
  });

  replaceBtn.addEventListener('click', function() {
    const selectedFile = videoFileInput.files[0];

    // Check if active tab is a YouTube watch page
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const tab = tabs[0];
      if (!tab.url || !tab.url.startsWith('https://www.youtube.com/watch')) {
        alert('Please open a YouTube watch page first.');
        return;
      }

      if (selectedFile) {
        // Use selected file
        const reader = new FileReader();
        reader.onload = function(e) {
          const arrayBuffer = e.target.result;
          const base64 = btoa(
            new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
          );

          // Send message to content script with video data
          chrome.tabs.sendMessage(tab.id, {
            action: 'replaceVideo',
            videoData: base64,
            mimeType: selectedFile.type
          }, handleResponse);
        };
        reader.readAsArrayBuffer(selectedFile);
      } else {
        // Try to use final composed video from backend if available
        chrome.runtime.sendMessage({ action: 'getAnalyzeState' }, (res) => {
          const state = (res && res.state) || {};
          const finalUrl = state.final_url;
          const payload = finalUrl ? { action: 'replaceVideo', serverUrl: finalUrl } : { action: 'replaceVideo' };
          chrome.tabs.sendMessage(tab.id, payload, handleResponse);
        });
      }
    });

    function handleResponse(response) {
      if (chrome.runtime.lastError) {
        alert('Error: Could not communicate with the page. Make sure the extension is loaded on YouTube.');
      } else if (response && response.success) {
        alert('Video replaced successfully! Check the browser console for debug info.');
      } else {
        alert('Failed to replace video. Check the browser console for error details.');
      }
    }
  });

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
      const res = analyze_state.result;
      if (res) {
        const result = res.result || JSON.stringify(res, null, 2);
        analyzeOutput.textContent += '\n\n' + result;
      }
      if (analyze_state.error) {
        analyzeOutput.textContent = 'Error: ' + analyze_state.error;
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
      const llm = (msg.llm || {});
      const result = llm.result || JSON.stringify(msg, null, 2);
      status['summary'] = 'completed';
      render();
      analyzeOutput.textContent += '\n\n' + result;
    } else if (msg.type === 'error') {
      analyzeOutput.textContent = 'Error: ' + msg.error;
    } else if (msg.type === 'final') {
      // Notify user final video is ready
      const vid = msg.video_id || '';
      const fileName = vid ? `final_${vid}.mp4` : (msg.final_path || '').split('/').pop() || 'final.mp4';
      analyzeOutput.textContent += `\n\n✅ Final video ready: ${fileName}.\nPress "gHost This" on the page to play it.`;
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
