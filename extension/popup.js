document.addEventListener('DOMContentLoaded', function() {
  const replaceBtn = document.getElementById('replaceBtn');
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
        // Use bundled video file
        chrome.tabs.sendMessage(tab.id, {
          action: 'replaceVideo'
        }, handleResponse);
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
});