(function () {
  const form = document.querySelector("[data-recorder-form]");
  if (!form) {
    return;
  }

  const startButton = form.querySelector("[data-record-start]");
  const stopButton = form.querySelector("[data-record-stop]");
  const playButton = form.querySelector("[data-record-play]");
  const deleteButton = form.querySelector("[data-record-delete]");
  const status = form.querySelector("[data-recording-status]");
  const fileInput = form.querySelector("input[type='file']");
  const preview = form.querySelector("[data-recording-preview]");
  const stage = form.querySelector("[data-recording-stage]");
  const waveformCard = form.querySelector("[data-waveform-card]");
  const waveformCanvas = form.querySelector("[data-waveform-canvas]");
  const waveformPlayhead = form.querySelector("[data-waveform-playhead]");
  const recordingTime = form.querySelector("[data-recording-time]");
  const recordingDuration = form.querySelector("[data-recording-duration]");
  const scriptSelect = form.querySelector("select[name='script']");
  const scriptSearch = form.querySelector("[data-script-search]");
  const randomScriptButton = form.querySelector("[data-random-script]");
  const previewBase = form.dataset.scriptPreviewBase;
  const previewPanel = document.querySelector("[data-script-preview-panel]");
  const scriptTitle = document.querySelector("[data-script-title]");
  const scriptMeta = document.querySelector("[data-script-meta]");
  const scriptWordCount = document.querySelector("[data-script-word-count]");
  const scriptBody = document.querySelector("[data-script-body]");

  let recorder = null;
  let chunks = [];
  let stream = null;
  let previewRequestId = 0;
  let audioContext = null;
  let analyser = null;
  let streamSource = null;
  let animationId = null;
  let playheadAnimationId = null;
  let recordingStartedAt = 0;
  let recordingUrl = null;
  let decodedBuffer = null;
  let renderedPeaks = [];

  const originalScriptOptions = scriptSelect
    ? Array.from(scriptSelect.options).map((option) => ({
        value: option.value,
        text: option.textContent,
      }))
    : [];

  function previewUrl(scriptId) {
    return previewBase.replace(/\/0\/preview\/?$/, `/${encodeURIComponent(scriptId)}/preview/`);
  }

  function renderBody(text) {
    scriptBody.replaceChildren();
    const blocks = text.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean);
    if (!blocks.length) {
      const paragraph = document.createElement("p");
      paragraph.textContent = "This script has no readable text yet.";
      scriptBody.appendChild(paragraph);
      return;
    }
    blocks.forEach((block) => {
      const paragraph = document.createElement("p");
      paragraph.textContent = block;
      scriptBody.appendChild(paragraph);
    });
  }

  function renderMeta(script) {
    const values = [
      script.author,
      script.source,
      `Level ${script.difficulty}`,
      ...(script.tags || []),
    ].filter(Boolean);
    scriptMeta.replaceChildren();
    values.forEach((value) => {
      const chip = document.createElement("span");
      chip.textContent = value;
      scriptMeta.appendChild(chip);
    });
  }

  async function updateScriptPreview() {
    if (!scriptSelect || !previewBase || !previewPanel) {
      return;
    }
    const scriptId = scriptSelect.value;
    if (!scriptId) {
      scriptTitle.textContent = "No active scripts yet";
      scriptWordCount.textContent = "No script selected";
      scriptMeta.replaceChildren();
      renderBody("Your practice text will appear here after a script is available.");
      return;
    }

    const requestId = ++previewRequestId;
    previewPanel.classList.add("is-loading");
    try {
      const response = await fetch(previewUrl(scriptId), {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error("Script preview failed.");
      }
      const script = await response.json();
      if (requestId !== previewRequestId) {
        return;
      }
      scriptTitle.textContent = script.title;
      scriptWordCount.textContent = `${script.word_count} words`;
      renderMeta(script);
      renderBody(script.body);
    } catch (error) {
      if (requestId === previewRequestId) {
        scriptWordCount.textContent = "Preview unavailable";
      }
    } finally {
      if (requestId === previewRequestId) {
        previewPanel.classList.remove("is-loading");
      }
    }
  }

  if (scriptSelect) {
    scriptSelect.addEventListener("change", updateScriptPreview);
  }

  function visibleScriptValues() {
    if (!scriptSelect) {
      return [];
    }
    return Array.from(scriptSelect.options).filter((option) => option.value).map((option) => option.value);
  }

  function filterScripts() {
    if (!scriptSelect || !scriptSearch) {
      return;
    }
    const query = scriptSearch.value.trim().toLowerCase();
    const selected = scriptSelect.value;
    scriptSelect.replaceChildren();
    originalScriptOptions.forEach((optionData) => {
      if (optionData.value && query && !optionData.text.toLowerCase().includes(query)) {
        return;
      }
      const option = document.createElement("option");
      option.value = optionData.value;
      option.textContent = optionData.text;
      scriptSelect.appendChild(option);
    });
    if (visibleScriptValues().includes(selected)) {
      scriptSelect.value = selected;
    } else if (visibleScriptValues().length) {
      scriptSelect.value = visibleScriptValues()[0];
      updateScriptPreview();
    }
  }

  if (scriptSearch) {
    scriptSearch.addEventListener("input", filterScripts);
  }

  if (randomScriptButton && scriptSelect) {
    randomScriptButton.addEventListener("click", function () {
      const values = visibleScriptValues();
      if (!values.length) {
        return;
      }
      const current = scriptSelect.value;
      let next = values[Math.floor(Math.random() * values.length)];
      if (values.length > 1) {
        while (next === current) {
          next = values[Math.floor(Math.random() * values.length)];
        }
      }
      scriptSelect.value = next;
      scriptSelect.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  function formatTime(seconds) {
    const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    const minutes = Math.floor(safeSeconds / 60);
    const remaining = Math.floor(safeSeconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
  }

  function getAudioContext() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      return null;
    }
    if (!audioContext) {
      audioContext = new AudioContextClass();
    }
    return audioContext;
  }

  function sizeCanvas() {
    if (!waveformCanvas) {
      return { width: 0, height: 0, ratio: 1 };
    }
    const ratio = window.devicePixelRatio || 1;
    const rect = waveformCanvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * ratio));
    const height = Math.max(1, Math.floor(rect.height * ratio));
    if (waveformCanvas.width !== width || waveformCanvas.height !== height) {
      waveformCanvas.width = width;
      waveformCanvas.height = height;
    }
    return { width, height, ratio };
  }

  function clearWaveform() {
    if (!waveformCanvas) {
      return;
    }
    const context = waveformCanvas.getContext("2d");
    const { width, height } = sizeCanvas();
    context.clearRect(0, 0, width, height);
    waveformCard.classList.remove("has-waveform");
    setPlayhead(0);
  }

  function setPlayhead(progress) {
    if (!waveformPlayhead) {
      return;
    }
    const percent = Math.max(0, Math.min(1, progress || 0)) * 100;
    waveformPlayhead.style.left = `${percent}%`;
  }

  function drawWaveform(values, progress) {
    if (!waveformCanvas || !values || !values.length) {
      return;
    }
    const context = waveformCanvas.getContext("2d");
    const { width, height } = sizeCanvas();
    const middle = height / 2;
    const barCount = Math.min(width, values.length);
    const stride = values.length / barCount;

    context.clearRect(0, 0, width, height);
    context.lineWidth = Math.max(1, window.devicePixelRatio || 1);
    context.strokeStyle = "rgba(169, 195, 191, 0.26)";
    context.beginPath();
    context.moveTo(0, middle);
    context.lineTo(width, middle);
    context.stroke();

    context.beginPath();
    context.strokeStyle = "#28d7e8";
    context.fillStyle = "rgba(40, 215, 232, 0.14)";
    context.moveTo(0, middle);
    for (let i = 0; i < barCount; i += 1) {
      const sample = Math.abs(values[Math.floor(i * stride)] || 0);
      const x = (i / Math.max(1, barCount - 1)) * width;
      const y = middle - sample * middle * 0.86;
      context.lineTo(x, y);
    }
    for (let i = barCount - 1; i >= 0; i -= 1) {
      const sample = Math.abs(values[Math.floor(i * stride)] || 0);
      const x = (i / Math.max(1, barCount - 1)) * width;
      const y = middle + sample * middle * 0.86;
      context.lineTo(x, y);
    }
    context.closePath();
    context.fill();

    context.beginPath();
    context.strokeStyle = "#52f0ff";
    for (let i = 0; i < barCount; i += 1) {
      const sample = Math.abs(values[Math.floor(i * stride)] || 0);
      const x = (i / Math.max(1, barCount - 1)) * width;
      const y = middle - sample * middle * 0.86;
      if (i === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
    context.stroke();

    waveformCard.classList.add("has-waveform");
    setPlayhead(progress || 0);
  }

  function buildPeaks(buffer, count) {
    const channel = buffer.getChannelData(0);
    const samplesPerPeak = Math.max(1, Math.floor(channel.length / count));
    const peaks = [];
    for (let i = 0; i < count; i += 1) {
      const start = i * samplesPerPeak;
      const end = Math.min(channel.length, start + samplesPerPeak);
      let max = 0;
      for (let j = start; j < end; j += 1) {
        const value = Math.abs(channel[j]);
        if (value > max) {
          max = value;
        }
      }
      peaks.push(max);
    }
    return peaks;
  }

  async function renderBlobWaveform(blob) {
    const context = getAudioContext();
    if (!context || !blob || !waveformCanvas) {
      return;
    }
    try {
      const buffer = await blob.arrayBuffer();
      decodedBuffer = await context.decodeAudioData(buffer.slice(0));
      renderedPeaks = buildPeaks(decodedBuffer, 1200);
      drawWaveform(renderedPeaks, 0);
      recordingTime.textContent = "00:00";
      recordingDuration.textContent = `${formatTime(decodedBuffer.duration)} take`;
      playButton.disabled = false;
    } catch (error) {
      recordingDuration.textContent = "waveform unavailable";
    }
  }

  function stopLiveWaveform() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    if (streamSource) {
      streamSource.disconnect();
      streamSource = null;
    }
    analyser = null;
  }

  function startLiveWaveform(mediaStream) {
    const context = getAudioContext();
    if (!context || !waveformCanvas) {
      return;
    }
    analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    streamSource = context.createMediaStreamSource(mediaStream);
    streamSource.connect(analyser);
    const samples = new Float32Array(analyser.fftSize);

    function tick() {
      analyser.getFloatTimeDomainData(samples);
      const elapsed = (Date.now() - recordingStartedAt) / 1000;
      recordingTime.textContent = formatTime(elapsed);
      recordingDuration.textContent = "recording";
      drawWaveform(samples, 1);
      animationId = requestAnimationFrame(tick);
    }

    tick();
  }

  function updatePlaybackState() {
    if (!preview || !decodedBuffer) {
      return;
    }
    const duration = preview.duration || decodedBuffer.duration || 0;
    const current = preview.currentTime || 0;
    const progress = duration ? current / duration : 0;
    recordingTime.textContent = formatTime(current);
    recordingDuration.textContent = `${formatTime(duration)} take`;
    if (renderedPeaks.length) {
      drawWaveform(renderedPeaks, progress);
    } else {
      setPlayhead(progress);
    }
  }

  function animatePlayhead() {
    updatePlaybackState();
    if (preview && !preview.paused && !preview.ended) {
      playheadAnimationId = requestAnimationFrame(animatePlayhead);
    }
  }

  function hasTake() {
    return Boolean(decodedBuffer || (fileInput && fileInput.files && fileInput.files.length));
  }

  function setRecording(active) {
    startButton.disabled = active;
    stopButton.disabled = !active;
    playButton.disabled = active || !decodedBuffer;
    if (deleteButton) {
      deleteButton.disabled = active || !hasTake();
    }
    stage.classList.toggle("is-recording", active);
    startButton.classList.toggle("is-recording", active);
    status.textContent = active ? "Recording..." : "Recording ready";
  }

  function loadBlobIntoPreview(blob, filename) {
    if (recordingUrl) {
      URL.revokeObjectURL(recordingUrl);
    }
    recordingUrl = URL.createObjectURL(blob);
    preview.src = recordingUrl;
    preview.hidden = true;
    if (filename && fileInput.files.length) {
      status.textContent = `Loaded ${filename}`;
    }
    renderBlobWaveform(blob);
    if (deleteButton) {
      deleteButton.disabled = false;
    }
  }

  function deleteCurrentTake() {
    if (preview) {
      preview.pause();
      preview.removeAttribute("src");
      preview.load();
    }
    if (recordingUrl) {
      URL.revokeObjectURL(recordingUrl);
      recordingUrl = null;
    }
    if (fileInput) {
      fileInput.value = "";
    }
    decodedBuffer = null;
    renderedPeaks = [];
    chunks = [];
    clearWaveform();
    recordingTime.textContent = "00:00";
    recordingDuration.textContent = "No take yet";
    status.textContent = "Ready to record";
    playButton.disabled = true;
    playButton.classList.remove("is-playing");
    playButton.setAttribute("aria-label", "Play recording");
    if (deleteButton) {
      deleteButton.disabled = true;
    }
  }

  if (waveformCanvas) {
    clearWaveform();
    window.addEventListener("resize", function () {
      if (renderedPeaks.length) {
        drawWaveform(renderedPeaks, preview && preview.duration ? preview.currentTime / preview.duration : 0);
      } else {
        clearWaveform();
      }
    });
    waveformCanvas.addEventListener("click", function (event) {
      if (!preview || !decodedBuffer || !preview.src) {
        return;
      }
      const rect = waveformCanvas.getBoundingClientRect();
      const progress = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      const duration = preview.duration || decodedBuffer.duration || 0;
      preview.currentTime = duration * progress;
      updatePlaybackState();
      preview.play();
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        return;
      }
      decodedBuffer = null;
      renderedPeaks = [];
      playButton.disabled = true;
      loadBlobIntoPreview(file, file.name);
    });
  }

  if (preview) {
    preview.addEventListener("play", function () {
      playButton.classList.add("is-playing");
      playButton.setAttribute("aria-label", "Pause recording");
      animatePlayhead();
    });
    preview.addEventListener("pause", function () {
      playButton.classList.remove("is-playing");
      playButton.setAttribute("aria-label", "Play recording");
      if (playheadAnimationId) {
        cancelAnimationFrame(playheadAnimationId);
        playheadAnimationId = null;
      }
      updatePlaybackState();
    });
    preview.addEventListener("ended", function () {
      playButton.classList.remove("is-playing");
      playButton.setAttribute("aria-label", "Play recording");
      setPlayhead(0);
      recordingTime.textContent = "00:00";
    });
  }

  if (playButton) {
    playButton.addEventListener("click", function () {
      if (!preview || !preview.src) {
        return;
      }
      if (preview.paused) {
        preview.play();
      } else {
        preview.pause();
      }
    });
  }

  if (deleteButton) {
    deleteButton.addEventListener("click", deleteCurrentTake);
  }

  if (!navigator.mediaDevices || !window.MediaRecorder) {
    if (startButton && status) {
      startButton.disabled = true;
      status.textContent = "Browser recording unavailable. Upload an audio file instead.";
    }
    return;
  }

  startButton.addEventListener("click", async function () {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      decodedBuffer = null;
      renderedPeaks = [];
      playButton.disabled = true;
      recordingStartedAt = Date.now();
      recorder = new MediaRecorder(stream);
      recorder.addEventListener("dataavailable", function (event) {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      });
      recorder.addEventListener("stop", function () {
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        const file = new File([blob], "browser-recording.webm", { type: blob.type });
        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInput.files = transfer.files;
        loadBlobIntoPreview(blob);
        stream.getTracks().forEach((track) => track.stop());
        stopLiveWaveform();
        setRecording(false);
      });
      recorder.start();
      setRecording(true);
      startLiveWaveform(stream);
    } catch (error) {
      status.textContent = "Microphone access failed. Use file upload instead.";
      setRecording(false);
      stopLiveWaveform();
    }
  });

  stopButton.addEventListener("click", function () {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  });
})();
