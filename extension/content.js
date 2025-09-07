// Function to replace video with provided source, or bundled local video
function replaceVideoWithLocal(src) {
  const videoSrc = src || chrome.runtime.getURL('local_video.mp4');
  console.log('🎥 gHost Mode activated');

  // Find and hide original video elements to stop YouTube playback
  const originalVideos = document.querySelectorAll('video');
  originalVideos.forEach(video => {
    video.pause(); // Pause first
    video.style.display = 'none';
  });

  // Stop YouTube player if it exists
  if (window.yt && window.yt.player && window.yt.player.getPlayerByElement) {
    try {
      const player = window.yt.player.getPlayerByElement(document.querySelector('.html5-video-player'));
      if (player && player.pauseVideo) {
        player.pauseVideo();
      }
    } catch (e) {
      console.log('Could not stop YouTube player:', e);
    }
  }

  // Find the player container
  const playerContainer = document.querySelector('#player') || document.querySelector('#ytd-player') || document.querySelector('.html5-video-player');
  if (playerContainer) {
    // Save original dimensions
    const rect = playerContainer.getBoundingClientRect();
    const originalWidth = rect.width || playerContainer.offsetWidth || 640;
    const originalHeight = rect.height || playerContainer.offsetHeight || 360;

    // Create a wrapper div to contain our video
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.width = '100%';
    wrapper.style.height = originalHeight + 'px';
    wrapper.style.overflow = 'visible';
    wrapper.style.zIndex = '9999';

    // Create new video element
    const video = document.createElement('video');
    video.width = originalWidth;
    video.height = originalHeight;
    video.controls = true;
    video.autoplay = true;
    video.preload = 'auto';
    video.muted = false; // Allow audio
    video.style.border = 'none';
    video.style.display = 'block';
    video.style.objectFit = 'contain';
    video.style.backgroundColor = '#000';
    video.style.position = 'relative';

    console.log('Video dimensions:', originalWidth, 'x', originalHeight);

    video.addEventListener('error', (e) => {
       console.error('Video error:', video.error ? video.error.code : 'unknown', video.error ? video.error.message : 'unknown', e);
     });
    video.addEventListener('loadedmetadata', () => {
      console.log('Video metadata loaded, duration:', video.duration);
    });
    video.addEventListener('loadeddata', () => {
      console.log('Video data loaded successfully');
    });
    video.addEventListener('canplay', () => {
      console.log('Video can play, attempting to play...');
      video.play().catch(e => {
        console.error('Play failed:', e);
        // Try with muted if autoplay fails
        video.muted = true;
        video.play().catch(e2 => console.error('Muted play also failed:', e2));
      });
    });
    video.addEventListener('play', () => {
      console.log('🎉 gHost Mode: Video started playing');
    });

    // Set video source
    video.src = videoSrc;
    video.load();

    // Add video to wrapper, then wrapper to container
    wrapper.appendChild(video);

    // Clear container and add our wrapper
    playerContainer.innerHTML = '';
    playerContainer.appendChild(wrapper);

    return true;
  } else {
    console.error('Player container not found');
    return false;
  }
}

// Function to create and inject the gHost This button in the actions row
function createGhostModeButton() {
  // Check if button already exists
  if (document.getElementById('ghost-mode-btn')) {
    console.log('👻 gHost Mode button already exists');
    return;
  }

  // Primary target: Actions row next to Like/Share/etc.
  const anchorEl = document.querySelector(
    '#actions-inner #top-level-buttons-computed, ' +
    '#actions #top-level-buttons-computed, ' +
    'ytd-menu-renderer#top-level-buttons-computed'
  );

  if (!anchorEl) {
    console.log('👻 Actions row not ready yet');
    return;
  }

  // Create the gHost This button
  const ghostButton = document.createElement('button');
  ghostButton.id = 'ghost-mode-btn';
  ghostButton.innerHTML = '👻 gHost This';
  ghostButton.title = 'Replace video with local/bundled video';

  // Style to match YouTube buttons
  ghostButton.style.cssText = `
    background: #ff0000;
    color: white;
    border: none;
    border-radius: 18px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    margin-left: 8px;
    transition: all 0.2s ease;
    font-family: 'Roboto', 'Arial', sans-serif;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    vertical-align: middle;
  `;

  // Disable initially; will be enabled when processing finishes
  const setDisabled = (btn, disabled) => {
    btn.disabled = disabled;
    if (disabled) {
      btn.style.background = '#666';
      btn.style.cursor = 'not-allowed';
      btn.title = 'Processing… please wait';
    } else {
      btn.style.background = '#ff0000';
      btn.style.cursor = 'pointer';
      btn.title = 'Replace video with composed final';
    }
  };
  setDisabled(ghostButton, true);

  // Add hover effects
  ghostButton.onmouseover = () => {
    ghostButton.style.background = '#cc0000';
    ghostButton.style.transform = 'scale(1.05)';
  };
  ghostButton.onmouseout = () => {
    ghostButton.style.background = '#ff0000';
    ghostButton.style.transform = 'scale(1)';
  };

  // Add click handler
  ghostButton.onclick = () => {
    if (ghostButton.disabled) {
      console.log('gHost button is disabled while processing');
      return;
    }
    console.log('👻 gHost Mode button clicked!');
    // Try to use final composed video from backend state; fallback to bundled
    chrome.runtime.sendMessage({ action: 'getAnalyzeState' }, async (res) => {
      const state = (res && res.state) || {};
      // If analyze state doesn't have a final URL yet, fall back to local backend route for DFnoQkYUqgU
      const finalUrl = state.final_url || 'http://127.0.0.1:8000/final_video/DFnoQkYUqgU';
      let success = false;
      if (finalUrl) {
        console.log('Fetching final video blob from:', finalUrl);
        try {
          const r = await fetch(finalUrl);
          if (!r.ok) throw new Error('HTTP ' + r.status);
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          success = replaceVideoWithLocal(url);
        } catch (e) {
          console.error('Failed fetching final video:', e);
        }
      }
      if (!success) {
        success = replaceVideoWithLocal();
      }
      if (success) {
        ghostButton.innerHTML = '✅ gHost This';
        ghostButton.style.background = '#00ff00';
        setTimeout(() => {
          ghostButton.innerHTML = '👻 gHost This';
          ghostButton.style.background = '#ff0000';
        }, 2000);
      } else {
        ghostButton.innerHTML = '❌ Failed';
        ghostButton.style.background = '#666';
        setTimeout(() => {
          ghostButton.innerHTML = '👻 gHost This';
          ghostButton.style.background = '#ff0000';
        }, 2000);
      }
    });
  };

  // Insert the button next to the Share button when possible
  try {
    let inserted = false;
    if (anchorEl) {
      const shareSelectors = [
        'button[aria-label*="Share" i]',
        'yt-button-shape[aria-label*="Share" i] button',
        'ytd-button-renderer[button-renderer] button[aria-label*="Share" i]',
        'tp-yt-paper-button[aria-label*="Share" i]'
      ];
      let shareButton = null;
      for (const s of shareSelectors) {
        const el = anchorEl.querySelector(s);
        if (el) { shareButton = el; break; }
      }
      if (shareButton && shareButton.insertAdjacentElement) {
        shareButton.insertAdjacentElement('afterend', ghostButton);
        inserted = true;
      }
      if (!inserted) {
        if (anchorEl.insertAdjacentElement) {
          anchorEl.insertAdjacentElement('beforeend', ghostButton);
          inserted = true;
        } else if (anchorEl.appendChild) {
          anchorEl.appendChild(ghostButton);
          inserted = true;
        }
      }
    }
    if (inserted) console.log('👻 gHost This button injected in actions row');

  // Verify visibility and fallback to floating button if needed
    const ensureVisible = () => {
      const btn = document.getElementById('ghost-mode-btn');
      if (!btn) return;
      const rect = btn.getBoundingClientRect();
      const styles = getComputedStyle(btn);
      const hidden = !rect || rect.width === 0 || rect.height === 0 ||
                     styles.display === 'none' || styles.visibility === 'hidden' ||
                     !btn.isConnected || btn.offsetParent === null;
      if (hidden) {
        console.log('⚠️ gHost button hidden in layout; creating floating fallback');
        if (typeof window.createFloatingGhostButton === 'function') {
          window.createFloatingGhostButton();
        }
      }
    };
    setTimeout(ensureVisible, 800);
    setTimeout(ensureVisible, 2000);
  } catch (error) {
    console.log('❌ Failed to insert gHost button:', error);
  }
}

// Function to wait for YouTube to load and inject button
function waitForYouTubeAndInject() {
  const observer = new MutationObserver(() => {
    if (!document.getElementById('ghost-mode-btn')) {
      createGhostModeButton();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });

  setTimeout(createGhostModeButton, 800);
  setTimeout(createGhostModeButton, 2000);
}

// Removed verbose debug/test helpers to keep the script lean

// Alternative injection method - floating button
window.createFloatingGhostButton = function() {
  // Remove existing button
  const existing = document.getElementById('ghost-mode-btn');
  if (existing) existing.remove();

  // Create floating button
  const ghostButton = document.createElement('button');
  ghostButton.id = 'ghost-mode-btn';
  ghostButton.innerHTML = '👻 gHost This';
  ghostButton.title = 'Replace video with local/bundled video';

  // Style as floating button
  ghostButton.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #ff0000;
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 20px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    z-index: 10000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
    font-family: 'Roboto', 'Arial', sans-serif;
  `;

  // Add hover effects
  ghostButton.onmouseover = () => {
    ghostButton.style.background = '#cc0000';
    ghostButton.style.transform = 'scale(1.05)';
  };
  ghostButton.onmouseout = () => {
    ghostButton.style.background = '#ff0000';
    ghostButton.style.transform = 'scale(1)';
  };

  // Add click handler
  // start disabled
  const setDisabled = (btn, disabled) => {
    btn.disabled = disabled;
    if (disabled) {
      btn.style.background = '#666';
      btn.style.cursor = 'not-allowed';
      btn.title = 'Processing… please wait';
    } else {
      btn.style.background = '#ff0000';
      btn.style.cursor = 'pointer';
      btn.title = 'Replace video with composed final';
    }
  };
  setDisabled(ghostButton, true);

  ghostButton.onclick = () => {
    if (ghostButton.disabled) {
      console.log('gHost button is disabled while processing');
      return;
    }
    console.log('👻 Floating gHost Mode button clicked!');
    chrome.runtime.sendMessage({ action: 'getAnalyzeState' }, async (res) => {
      const state = (res && res.state) || {};
      // If analyze state doesn't have a final URL yet, fall back to local backend route for DFnoQkYUqgU
      const finalUrl = state.final_url || 'http://127.0.0.1:8000/final_video/DFnoQkYUqgU';
      let success = false;
      if (finalUrl) {
        console.log('Fetching final video blob from:', finalUrl);
        try {
          const r = await fetch(finalUrl);
          if (!r.ok) throw new Error('HTTP ' + r.status);
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          success = replaceVideoWithLocal(url);
        } catch (e) {
          console.error('Failed fetching final video:', e);
        }
      }
      if (!success) success = replaceVideoWithLocal();
      if (success) {
        ghostButton.innerHTML = '✅ gHost This';
        ghostButton.style.background = '#00ff00';
        setTimeout(() => {
          ghostButton.innerHTML = '👻 gHost This';
          ghostButton.style.background = '#ff0000';
        }, 2000);
      } else {
        ghostButton.innerHTML = '❌ Failed';
        ghostButton.style.background = '#666';
        setTimeout(() => {
          ghostButton.innerHTML = '👻 gHost This';
          ghostButton.style.background = '#ff0000';
        }, 2000);
      }
    });
  };

  // Add to page
  document.body.appendChild(ghostButton);
  console.log('🎯 Floating gHost Mode button created!');
};

// Enable the gHost button when a final video is ready or on restore
function enableGhostButtonIfReady() {
  chrome.runtime.sendMessage({ action: 'getAnalyzeState' }, (res) => {
    const state = (res && res.state) || {};
    if (state && state.final_url) {
      const btn = document.getElementById('ghost-mode-btn');
      if (btn) {
        btn.disabled = false;
        btn.style.background = '#ff0000';
        btn.style.cursor = 'pointer';
        btn.title = 'Replace video with composed final';
      }
    }
  });
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === 'final') {
    const btn = document.getElementById('ghost-mode-btn');
    if (btn) {
      btn.disabled = false;
      btn.style.background = '#ff0000';
      btn.style.cursor = 'pointer';
      btn.title = 'Replace video with composed final';
    }
  }
});

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { waitForYouTubeAndInject(); enableGhostButtonIfReady(); });
} else {
  waitForYouTubeAndInject();
  enableGhostButtonIfReady();
}

// One more delayed attempt just in case
setTimeout(() => {
  if (!document.getElementById('ghost-mode-btn')) createGhostModeButton();
  enableGhostButtonIfReady();
}, 3000);

// Also listen for navigation changes (YouTube SPA)
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    setTimeout(() => {
      if (!document.getElementById('ghost-mode-btn')) {
        createGhostModeButton();
      }
    }, 2000);
  }
}).observe(document, { subtree: true, childList: true });

// Listen for messages from popup (existing functionality)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'replaceVideo') {
    // If serverUrl provided, fetch and use blob URL to avoid mixed content issues
    if (request.serverUrl) {
      (async () => {
        try {
          console.log('Fetching server video for replacement:', request.serverUrl);
          const r = await fetch(request.serverUrl);
          if (!r.ok) throw new Error('HTTP ' + r.status);
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          const success = replaceVideoWithLocal(url);
          sendResponse({ success });
        } catch (e) {
          console.error('Failed to fetch server video:', e);
          const success = replaceVideoWithLocal();
          sendResponse({ success, error: String(e) });
        }
      })();
      return true; // keep message channel open for async response
    }
    // Fallbacks: use base64 data or packaged video
    let videoSrc;
    if (request.videoData && request.mimeType) {
      videoSrc = `data:${request.mimeType};base64,${request.videoData}`;
      console.log('Using user-selected video file');
    } else {
      videoSrc = chrome.runtime.getURL('local_video.mp4');
      console.log('Using bundled local video');
    }
    const success = replaceVideoWithLocal(videoSrc);
    sendResponse({ success });
  } else {
    sendResponse({ success: false, error: 'Invalid request' });
  }
});
