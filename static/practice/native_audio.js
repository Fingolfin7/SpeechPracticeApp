(() => {
  const record = document.getElementById('record');
  const stop = document.getElementById('stop');
  const audio = document.getElementById('audio');
  const preview = document.getElementById('preview');
  const status = document.getElementById('record-status');
  const form = document.getElementById('native-audio-form');
  let recorder, stream, timer, url;
  if (!navigator.mediaDevices || !window.MediaRecorder || !window.DataTransfer) {
    record.disabled = true;
    status.textContent = 'Recording is unavailable in this browser; choose an audio file instead.';
    return;
  }
  record.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: true});
      recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        clearTimeout(timer);
        stream.getTracks().forEach(track => track.stop());
        const blob = new Blob(chunks, {type: recorder.mimeType || 'audio/webm'});
        const files = new DataTransfer();
        files.items.add(new File([blob], 'recording', {type: blob.type}));
        audio.files = files.files;
        if (url) URL.revokeObjectURL(url);
        url = URL.createObjectURL(blob);
        preview.src = url;
        preview.hidden = false;
        record.disabled = false;
        stop.disabled = true;
        status.textContent = 'Recording ready. Listen before saving.';
      };
      recorder.start();
      record.disabled = true;
      stop.disabled = false;
      status.textContent = 'Recording… It will stop after 118 seconds to allow for encoding.';
      timer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 118000);
    } catch (_) {
      if (stream) stream.getTracks().forEach(track => track.stop());
      status.textContent = 'Microphone unavailable. Check your browser permission or choose an audio file.';
    }
  });
  stop.addEventListener('click', () => { if (recorder && recorder.state === 'recording') recorder.stop(); });
  form.addEventListener('submit', event => {
    if (recorder && recorder.state === 'recording') { event.preventDefault(); status.textContent = 'Stop the recording before saving.'; }
  });
})();
