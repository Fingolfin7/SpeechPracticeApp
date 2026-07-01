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
  const modeInputs = Array.from(form.querySelectorAll("input[name='mode']"));
  const modeHelper = form.querySelector("[data-mode-helper]");
  const scriptSearch = form.querySelector("[data-script-search]");
  const randomScriptButton = form.querySelector("[data-random-script]");
  const previewBase = form.dataset.scriptPreviewBase;
  const previewPanel = document.querySelector("[data-script-preview-panel]");
  const scriptTitle = document.querySelector("[data-script-title]");
  const scriptMeta = document.querySelector("[data-script-meta]");
  const scriptWordCount = document.querySelector("[data-script-word-count]");
  const scriptBody = document.querySelector("[data-script-body]");
  const ladderSelect = document.querySelector("[data-ladder-select]");
  const scoreButton = form.querySelector("[data-score-button]");
  const scoreReason = form.querySelector("[data-score-reason]");
  const copyScoreButton = form.querySelector("[data-copy-practice-score]");
  const autumnButton = form.querySelector("[data-autumn-toggle]");
  const autumnForm = document.querySelector("#autumn-timer-form");
  const transcriptState = form.querySelector("[data-transcript-state]");
  const liveTranscript = form.querySelector("[data-live-transcript]");
  const practiceMetrics = form.querySelector("[data-practice-metrics]");

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
  let activeTranscriptSegment = null;
  let activeJobStatusUrl = "";
  let jobPollTimeout = null;
  let latestScoreText = "";
  let submissionId = "";
  let jobPollDelay = 2500;
  let lastJobSignature = "";

  function newSubmissionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (token) {
      const random = Math.floor(Math.random() * 16);
      const value = token === "x" ? random : (random & 0x3) | 0x8;
      return value.toString(16);
    });
  }

  function resetSubmissionId() {
    submissionId = "";
  }

  function createSpeechRecorder(mediaStream) {
    try {
      return new MediaRecorder(mediaStream, { audioBitsPerSecond: 64000 });
    } catch (error) {
      return new MediaRecorder(mediaStream);
    }
  }

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
      script.practice_kind,
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

  function currentMode() {
    const checked = modeInputs.find((input) => input.checked);
    return checked ? checked.value : "script";
  }

  function updateModeUi(navigate) {
    const mode = currentMode();
    form.classList.toggle("is-free-speak", mode === "free_speak");
    form.classList.toggle("is-quick-practice", mode === "quick");
    if (modeHelper) {
      if (mode === "free_speak") {
        modeHelper.textContent = "Free Speak transcribes without scoring or updating cards.";
      } else if (mode === "quick") {
        modeHelper.textContent = "Quick Practice scores focused drills from your priority cards.";
      } else {
        modeHelper.textContent = "Script mode scores your recording against the selected text.";
      }
    }
    if (navigate && !hasTake()) {
      const url = new URL(window.location.href);
      if ((url.searchParams.get("mode") || "script") !== mode) {
        url.searchParams.set("mode", mode);
        url.searchParams.delete("script");
        url.searchParams.delete("card");
        window.location.assign(url.toString());
      }
    }
  }

  modeInputs.forEach((input) => {
    input.addEventListener("change", function () {
      updateModeUi(true);
    });
  });

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

  if (ladderSelect) {
    ladderSelect.addEventListener("change", function () {
      if (ladderSelect.value) {
        window.location.assign(ladderSelect.value);
      }
    });
  }

  updateModeUi(false);

  function formatTime(seconds) {
    const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    const minutes = Math.floor(safeSeconds / 60);
    const remaining = Math.floor(safeSeconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
  }

  function setTranscriptState(text) {
    if (transcriptState) {
      transcriptState.textContent = text;
    }
  }

  async function copyTextToClipboard(text) {
    if (!text) {
      return false;
    }
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied;
  }

  function setLiveTranscriptText(text) {
    if (!liveTranscript) {
      return;
    }
    liveTranscript.classList.remove("timed-transcript", "highlighted-reader");
    liveTranscript.textContent = text || "";
    activeTranscriptSegment = null;
  }

  function setLiveTranscriptHtml(html) {
    if (!liveTranscript) {
      return;
    }
    liveTranscript.classList.add("timed-transcript", "highlighted-reader");
    liveTranscript.innerHTML = html || '<span class="empty">No transcript yet.</span>';
    activeTranscriptSegment = null;
    setActiveTranscriptSegment();
  }

  function renderPracticeMetrics(metrics) {
    if (!practiceMetrics || !metrics) {
      return;
    }
    const values = [
      { tag: "strong", text: `Score: ${metrics.score || "-"}` },
      { tag: "span", text: `WER: ${metrics.wer || "-"}` },
      { tag: "span", text: `CER: ${metrics.cer || "-"}` },
      { tag: "span", text: `Clarity: ${metrics.clarity || "-"}` },
      { tag: "span", text: `Rate: ${metrics.artic_rate || "- wpm"}` },
      { tag: "span", text: `Pauses: ${metrics.pause_ratio || "-"}` },
      { tag: "span", text: `Conf: ${metrics.avg_conf || "-"}` },
    ];
    practiceMetrics.replaceChildren(
      ...values.map((item) => {
        const element = document.createElement(item.tag);
        element.textContent = item.text;
        return element;
      })
    );
  }

  function buildScoreText(metrics) {
    if (!metrics) {
      return "";
    }
    return [
      `Score: ${metrics.score || "-"}/5`,
      `WER: ${metrics.wer || "-"}`,
      `CER: ${metrics.cer || "-"}`,
      `Clarity: ${metrics.clarity || "-"}`,
      `Rate: ${metrics.artic_rate || "- wpm"}`,
      `Pauses: ${metrics.pause_ratio || "-"}`,
      `Conf: ${metrics.avg_conf || "-"}`,
    ].join(" | ");
  }

  function setCopyScoreEnabled(enabled) {
    if (!copyScoreButton) {
      return;
    }
    copyScoreButton.disabled = !enabled;
    if (!enabled) {
      copyScoreButton.textContent = "Copy score";
    }
  }

  function setScoreReady() {
    if (!scoreButton) {
      return;
    }
    const ready = hasTake();
    scoreButton.disabled = !ready;
    scoreButton.classList.toggle("is-disabled", !ready);
    if (scoreReason) {
      scoreReason.textContent = ready
        ? "Ready to score this take."
        : "Record or upload a take before scoring.";
    }
  }

  function setAutumnButtonBusy(busy) {
    if (!autumnButton) {
      return;
    }
    autumnButton.disabled = busy;
    if (busy) {
      autumnButton.dataset.previousText = autumnButton.textContent.trim();
      autumnButton.textContent = "Updating...";
    } else if (autumnButton.dataset.previousText) {
      autumnButton.textContent = autumnButton.dataset.previousText;
      delete autumnButton.dataset.previousText;
    }
  }

  function updateAutumnButton(payload) {
    if (!autumnButton || !payload) {
      return;
    }
    const active = Boolean(payload.active);
    autumnButton.disabled = false;
    autumnButton.classList.toggle("is-active", active);
    autumnButton.name = payload.button_name || (active ? "stop_autumn_timer" : "start_autumn_timer");
    autumnButton.textContent = payload.button_label || (active ? "Stop Autumn" : "Start Autumn");
    delete autumnButton.dataset.previousText;
  }

  function timedTranscriptSegments() {
    if (!liveTranscript) {
      return [];
    }
    return Array.from(liveTranscript.querySelectorAll("[data-start][data-end]"));
  }

  function setActiveTranscriptSegment() {
    const segments = timedTranscriptSegments();
    if (!segments.length || !preview) {
      if (activeTranscriptSegment) {
        activeTranscriptSegment.classList.remove("is-active");
        activeTranscriptSegment = null;
      }
      return;
    }
    const current = preview.currentTime || 0;
    const next = segments.find((segment, index) => {
      const start = Number(segment.dataset.start || 0);
      const end = Number(segment.dataset.end || start);
      const isLast = index === segments.length - 1;
      return current >= start && (current < end || (isLast && current <= end + 0.05));
    }) || segments.find((segment) => {
      const start = Number(segment.dataset.start || 0);
      const end = Number(segment.dataset.end || start);
      return current >= start - 0.05 && current <= end + 0.05;
    });
    if (next === activeTranscriptSegment) {
      return;
    }
    if (activeTranscriptSegment) {
      activeTranscriptSegment.classList.remove("is-active");
    }
    activeTranscriptSegment = next || null;
    if (activeTranscriptSegment) {
      activeTranscriptSegment.classList.add("is-active");
      activeTranscriptSegment.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }

  function closestTimedSegment(target) {
    if (!target) {
      return null;
    }
    const element = target instanceof Element ? target : target.parentElement;
    return element ? element.closest("[data-start]") : null;
  }

  function seekPreviewTo(targetTime) {
    if (!preview || !preview.src) {
      return;
    }
    const safeTime = Math.max(0, Number(targetTime) || 0);
    function applySeek() {
      preview.currentTime = safeTime;
      updatePlaybackState();
      setActiveTranscriptSegment();
      const playPromise = preview.play();
      if (playPromise) {
        playPromise.catch(function () {
          updatePlaybackState();
        });
      }
    }

    if (preview.readyState < 1) {
      preview.addEventListener("loadedmetadata", applySeek, { once: true });
      preview.load();
      return;
    }
    applySeek();
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

  function resumeAudioContextFromUserGesture() {
    const context = getAudioContext();
    if (!context || context.state !== "suspended") {
      return Promise.resolve(context);
    }
    return context.resume().then(function () {
      return context;
    }).catch(function () {
      return context;
    });
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
    setActiveTranscriptSegment();
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
    setScoreReady();
  }

  function loadBlobIntoPreview(blob, filename) {
    resetSubmissionId();
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
    setScoreReady();
  }

  function deleteCurrentTake() {
    resetSubmissionId();
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
    setScoreReady();
  }

  function setScoreButtonBusy(busy) {
    if (!scoreButton) {
      return;
    }
    scoreButton.disabled = busy || !hasTake();
    scoreButton.textContent = busy ? "Scoring..." : "Score recording";
    if (scoreReason) {
      scoreReason.textContent = busy
        ? "Scoring is running for this take."
        : hasTake()
          ? "Ready to score this take."
          : "Record or upload a take before scoring.";
    }
  }

  function renderPracticeJobStatus(payload) {
    if (!payload) {
      return;
    }
    if (payload.is_pending) {
      setTranscriptState(payload.partial_transcript ? "Transcribing" : payload.status_label || "Queued");
      if (payload.partial_transcript) {
        setLiveTranscriptText(payload.partial_transcript);
      }
      if (status) {
        status.textContent = payload.status_label === "Running" ? "Transcribing..." : "Scoring queued";
      }
      return;
    }

    if (payload.is_done) {
      setTranscriptState("Finished");
      if (payload.transcript_html) {
        setLiveTranscriptHtml(payload.transcript_html);
      } else if (payload.partial_transcript) {
        setLiveTranscriptText(payload.partial_transcript);
      }
      renderPracticeMetrics(payload.metrics);
      latestScoreText = payload.score_text || buildScoreText(payload.metrics);
      setCopyScoreEnabled(Boolean(latestScoreText));
      if (status) {
        status.textContent = "Scoring finished";
      }
      setScoreButtonBusy(false);
      return;
    }

    if (payload.is_failed) {
      setTranscriptState("Failed");
      setLiveTranscriptText(payload.error_message || "Scoring failed.");
      if (status) {
        status.textContent = "Scoring failed";
      }
      setScoreButtonBusy(false);
    }
  }

  async function pollPracticeJobStatus() {
    if (!activeJobStatusUrl) {
      return;
    }
    try {
      const response = await fetch(activeJobStatusUrl, {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error("Job status failed.");
      }
      const payload = await response.json();
      const signature = `${payload.status}:${payload.partial_transcript || ""}`;
      jobPollDelay = signature === lastJobSignature
        ? Math.min(10000, jobPollDelay * 1.5)
        : 2500;
      lastJobSignature = signature;
      renderPracticeJobStatus(payload);
      if (payload.is_pending) {
        const delay = document.hidden || !navigator.onLine ? 10000 : jobPollDelay;
        jobPollTimeout = window.setTimeout(pollPracticeJobStatus, delay);
      } else {
        activeJobStatusUrl = "";
        jobPollTimeout = null;
      }
    } catch (error) {
      setTranscriptState("Status unavailable");
      jobPollDelay = 10000;
      jobPollTimeout = window.setTimeout(pollPracticeJobStatus, jobPollDelay);
    }
  }

  function startPracticeJobPolling(statusUrl) {
    activeJobStatusUrl = statusUrl || "";
    jobPollDelay = 2500;
    lastJobSignature = "";
    if (jobPollTimeout) {
      window.clearTimeout(jobPollTimeout);
      jobPollTimeout = null;
    }
    if (activeJobStatusUrl) {
      pollPracticeJobStatus();
    }
  }

  async function submitPracticeForScoring(event) {
    event.preventDefault();
    if (!hasTake()) {
      setTranscriptState("Needs audio");
      setLiveTranscriptText("Record or upload audio before scoring.");
      if (status) {
        status.textContent = "Record or upload audio before scoring.";
      }
      setScoreReady();
      return;
    }
    if (jobPollTimeout) {
      window.clearTimeout(jobPollTimeout);
      jobPollTimeout = null;
    }
    latestScoreText = "";
    setCopyScoreEnabled(false);
    setScoreButtonBusy(true);
    setTranscriptState("Queued");
    setLiveTranscriptText("");
    if (status) {
      status.textContent = "Queueing scoring job...";
    }

    try {
      submissionId = submissionId || newSubmissionId();
      const response = await fetch(form.action || window.location.href, {
        method: "POST",
        body: new FormData(form),
        headers: {
          Accept: "application/json",
          "X-Requested-With": "XMLHttpRequest",
          "X-Idempotency-Key": submissionId,
        },
        credentials: "same-origin",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Scoring could not be queued.");
      }
      if (status && payload.message) {
        status.textContent = payload.message;
      }
      renderPracticeJobStatus(payload);
      startPracticeJobPolling(payload.status_url);
    } catch (error) {
      setTranscriptState("Needs audio");
      setLiveTranscriptText(error.message || "Record or upload audio before scoring.");
      if (status) {
        status.textContent = error.message || "Scoring could not be queued.";
      }
      setScoreButtonBusy(false);
    }
  }

  form.addEventListener("submit", submitPracticeForScoring);

  if (autumnButton && autumnForm) {
    autumnButton.addEventListener("click", async function (event) {
      event.preventDefault();
      const payload = new FormData(autumnForm);
      payload.set(autumnButton.name, autumnButton.value || "1");
      setAutumnButtonBusy(true);
      try {
        const response = await fetch(autumnForm.action, {
          method: "POST",
          body: payload,
          headers: {
            Accept: "application/json",
            "X-Requested-With": "XMLHttpRequest",
          },
          credentials: "same-origin",
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.ok) {
          throw new Error(data.message || "Autumn timer update failed.");
        }
        updateAutumnButton(data);
        if (status && data.message) {
          status.textContent = data.message;
        }
      } catch (error) {
        if (status) {
          status.textContent = error.message || "Autumn timer update failed.";
        }
        setAutumnButtonBusy(false);
      }
    });
  }

  if (copyScoreButton) {
    copyScoreButton.addEventListener("click", async function () {
      if (!latestScoreText) {
        return;
      }
      try {
        const copied = await copyTextToClipboard(latestScoreText);
        if (copied) {
          copyScoreButton.textContent = "Copied";
          window.setTimeout(function () {
            copyScoreButton.textContent = "Copy score";
          }, 1300);
        }
      } catch (error) {
        if (status) {
          status.textContent = latestScoreText;
        }
      }
    });
  }

  if (liveTranscript) {
    liveTranscript.addEventListener("click", function (event) {
      const segment = closestTimedSegment(event.target);
      if (segment) {
        seekPreviewTo(segment.dataset.start);
      }
    });
    liveTranscript.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      const segment = closestTimedSegment(event.target);
      if (!segment) {
        return;
      }
      event.preventDefault();
      seekPreviewTo(segment.dataset.start);
    });
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
        setScoreReady();
        return;
      }
      decodedBuffer = null;
      renderedPeaks = [];
      playButton.disabled = true;
      loadBlobIntoPreview(file, file.name);
      setScoreReady();
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
    preview.addEventListener("timeupdate", setActiveTranscriptSegment);
    preview.addEventListener("loadedmetadata", setActiveTranscriptSegment);
  }

  if (playButton) {
    playButton.addEventListener("click", function () {
      if (!preview || !preview.src) {
        return;
      }
      if (preview.paused) {
        preview.muted = false;
        preview.volume = 1;
        preview.play();
      } else {
        preview.pause();
      }
    });
  }

  if (deleteButton) {
    deleteButton.addEventListener("click", deleteCurrentTake);
  }

  function recorderUnavailableMessage() {
    if (!window.isSecureContext) {
      return {
        status: "Recording needs HTTPS on LAN.",
        helper: "Open the HTTPS address from run_https_server.bat, or upload audio instead.",
      };
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return {
        status: "Microphone API unavailable.",
        helper: "Try Chrome or Edge over HTTPS, or upload audio instead.",
      };
    }
    if (!window.MediaRecorder) {
      return {
        status: "Browser recording unavailable.",
        helper: "This browser cannot encode recordings here. Upload audio instead.",
      };
    }
    return null;
  }

  const unavailable = recorderUnavailableMessage();
  if (unavailable) {
    if (startButton) {
      startButton.disabled = true;
    }
    if (stage) {
      stage.classList.add("has-warning");
    }
    if (status) {
      status.textContent = unavailable.status;
    }
    if (modeHelper) {
      modeHelper.textContent = unavailable.helper;
    }
    setScoreReady();
    return;
  }

  setScoreReady();

  startButton.addEventListener("click", async function () {
    // Resume while this code still runs inside the click gesture. Chromium may
    // otherwise leave the analyser suspended after getUserMedia resolves,
    // producing a flat live waveform even though MediaRecorder is active.
    const audioContextReady = resumeAudioContextFromUserGesture();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      await audioContextReady;
      chunks = [];
      decodedBuffer = null;
      renderedPeaks = [];
      playButton.disabled = true;
      recordingStartedAt = Date.now();
      recorder = createSpeechRecorder(stream);
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
      setRecording(false);
      stopLiveWaveform();
      status.textContent = "Microphone access failed. Check browser and Windows input permissions.";
    }
  });

  stopButton.addEventListener("click", function () {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  });
})();
