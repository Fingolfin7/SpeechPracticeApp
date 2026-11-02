(() => {
  const form = document.getElementById('practice-form');
  const record = document.getElementById('record');
  const stop = document.getElementById('stop');
  const send = document.getElementById('send');
  const recorderArea = document.getElementById('recorder');
  const title = document.getElementById('record-title');
  const status = document.getElementById('record-status');
  const preview = document.getElementById('preview');
  const playback = document.getElementById('playback');
  const empty = document.getElementById('feedback-empty');
  const result = document.getElementById('feedback-result');
  let recorder, stream, blob, url, timer, busy = false;
  function showMessage(message) {
    empty.hidden = true; result.hidden = false; result.replaceChildren();
    const p = document.createElement('p'); p.textContent = message; result.append(p);
  }
  function renderFeedback(data) {
    empty.hidden = true; result.hidden = false; result.replaceChildren();
    const addText = (tag, text) => { const node = document.createElement(tag); node.textContent = text; result.append(node); };
    const addList = (heading, items) => {
      if (!items || !items.length) return;
      addText('h3', heading);
      const ul = document.createElement('ul');
      items.forEach(item => { const li = document.createElement('li'); li.textContent = item; ul.append(li); });
      result.append(ul);
    };
    if (data.status === 'ok') {
      addText('p', data.summary); addList('What worked', data.strengths); addList('What I noticed', data.observations); addList('Try next', data.practice);
      if (data.uncertainty) { addText('h3', 'Uncertainty'); addText('p', data.uncertainty); }
    } else if (data.status === 'unstructured') {
      addText('h3', 'Unverified response'); addText('p', data.text || 'The response was empty. Try again.');
    } else { addText('p', data.reason || 'Feedback was not available. Try again.'); }
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    record.disabled = true; status.textContent = 'Microphone recording is not available in this browser.'; return;
  }
  record.addEventListener('click', async () => {
    if (busy) return;
    record.disabled = true; send.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        clearTimeout(timer); stream.getTracks().forEach(track => track.stop());
        blob = new Blob(chunks, {type: recorder.mimeType || 'audio/webm'});
        if (url) URL.revokeObjectURL(url);
        url = URL.createObjectURL(blob); preview.src = url; playback.hidden = false;
        recorderArea.classList.remove('recording'); title.textContent = 'Recording ready';
        status.textContent = 'Listen back or send it for feedback.';
        record.textContent = 'Record again'; record.disabled = false; stop.disabled = true; send.disabled = !blob.size;
      };
      recorder.start(); blob = null; playback.hidden = true; record.disabled = true; stop.disabled = false; send.disabled = true;
      recorderArea.classList.add('recording'); title.textContent = 'Recording';
      status.textContent = 'Speak naturally. Recording stops automatically after 118 seconds.';
      timer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 118000);
    } catch (_) {
      if (stream) stream.getTracks().forEach(track => track.stop());
      record.disabled = false; send.disabled = !blob;
      status.textContent = 'Microphone unavailable. Check your browser permission and try again.';
    }
  });
  stop.addEventListener('click', () => { if (recorder && recorder.state === 'recording') recorder.stop(); });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!blob || busy || !form.reportValidity()) return;
    busy = true; record.disabled = true; stop.disabled = true; send.disabled = true; send.textContent = 'Listening…';
    showMessage('Listening to your recording. This may take a little while.');
    if (window.matchMedia('(max-width: 720px)').matches) result.scrollIntoView({block: 'start'});
    const data = new FormData(form); data.append('audio', blob, 'recording');
    try {
      const response = await fetch(form.action, {method: 'POST', body: data, credentials: 'same-origin'});
      renderFeedback(await response.json());
    } catch (_) {
      showMessage('The connection was interrupted. The request may have reached OpenAI and incurred a charge; check before sending again.');
    } finally {
      busy = false; record.disabled = false; send.disabled = !blob; send.textContent = 'Send for feedback';
    }
  });
})();
