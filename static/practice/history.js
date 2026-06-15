(function () {
  const player = document.querySelector("[data-history-player]");
  if (!player) {
    return;
  }

  const audio = player.querySelector("[data-history-audio]");
  const canvas = player.querySelector("[data-waveform]");
  const timedTranscript = document.querySelector("[data-timed-transcript]");
  if (!audio || !canvas) {
    return;
  }

  const ctx = canvas.getContext("2d");
  let waveformBuffer = null;
  let audioContext = null;
  let activeSegment = null;
  canvas.dataset.seekReady = "true";

  function drawEmpty(message) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#6f665b";
    ctx.font = "18px Georgia, serif";
    ctx.fillText(message, 24, canvas.height / 2);
  }

  function drawWaveform(buffer) {
    const data = buffer.getChannelData(0);
    const width = canvas.width;
    const height = canvas.height;
    const mid = height / 2;
    const step = Math.max(1, Math.floor(data.length / width));

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fffaf0";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 1;

    for (let x = 0; x < width; x += 1) {
      let min = 1;
      let max = -1;
      const start = x * step;
      const end = Math.min(start + step, data.length);
      for (let i = start; i < end; i += 1) {
        const value = data[i];
        if (value < min) min = value;
        if (value > max) max = value;
      }
      ctx.beginPath();
      ctx.moveTo(x, mid + min * mid * 0.84);
      ctx.lineTo(x, mid + max * mid * 0.84);
      ctx.stroke();
    }
  }

  function drawPlayhead() {
    if (!audio.duration || Number.isNaN(audio.duration)) {
      return;
    }
    const x = (audio.currentTime / audio.duration) * canvas.width;
    ctx.fillStyle = "rgba(184, 68, 33, 0.82)";
    ctx.fillRect(Math.max(0, x - 1), 0, 3, canvas.height);
  }

  function setActiveTranscriptSegment() {
    if (!timedTranscript) {
      return;
    }
    const current = audio.currentTime || 0;
    const next = Array.from(timedTranscript.querySelectorAll("[data-start][data-end]")).find((segment) => {
      const start = Number(segment.dataset.start || 0);
      const end = Number(segment.dataset.end || start);
      return current >= start - 0.05 && current <= end + 0.05;
    });
    if (next === activeSegment) {
      return;
    }
    if (activeSegment) {
      activeSegment.classList.remove("is-active");
    }
    activeSegment = next || null;
    if (activeSegment) {
      activeSegment.classList.add("is-active");
      activeSegment.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }

  async function loadWaveform() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      drawEmpty("Waveform unavailable. Audio controls still work.");
      return;
    }

    try {
      audioContext = new AudioContextClass();
      drawEmpty("Loading waveform...");
      const response = await fetch(audio.currentSrc || audio.src);
      const bytes = await response.arrayBuffer();
      const buffer = await audioContext.decodeAudioData(bytes);
      waveformBuffer = buffer;
      drawWaveform(buffer);
      audio.addEventListener("timeupdate", function () {
        drawWaveform(waveformBuffer);
        drawPlayhead();
        setActiveTranscriptSegment();
      });
    } catch (error) {
      drawEmpty("Waveform unavailable. Audio controls still work.");
    }
  }

  function seekAndPlay(event) {
    canvas.dataset.seekCount = String(Number(canvas.dataset.seekCount || "0") + 1);
    if (!audio.duration || Number.isNaN(audio.duration)) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const localX = typeof event.offsetX === "number" ? event.offsetX : event.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, localX / rect.width));
    const targetTime = Math.max(0, Math.min(audio.duration, ratio * audio.duration));
    canvas.dataset.seekTarget = targetTime.toFixed(3);
    audio.currentTime = targetTime;
    if (audioContext && audioContext.state === "suspended") {
      audioContext.resume();
    }
    if (waveformBuffer) {
      drawWaveform(waveformBuffer);
      drawPlayhead();
    }
    canvas.dataset.playState = "requested";
    const playPromise = audio.play();
    if (playPromise) {
      playPromise.then(function () {
        canvas.dataset.playState = audio.paused ? "paused-after-play" : "playing";
      }).catch(function (error) {
        canvas.dataset.playState = "blocked";
        canvas.dataset.playError = error.name || "playback-error";
        drawEmpty("Click the audio controls to allow playback.");
      });
    }
  }

  canvas.addEventListener("pointerdown", seekAndPlay);
  canvas.addEventListener("click", seekAndPlay);

  if (timedTranscript) {
    timedTranscript.addEventListener("click", function (event) {
      const segment = event.target.closest("[data-start]");
      if (!segment || !audio.duration || Number.isNaN(audio.duration)) {
        return;
      }
      audio.currentTime = Number(segment.dataset.start || 0);
      setActiveTranscriptSegment();
      audio.play();
    });
    timedTranscript.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      const segment = event.target.closest("[data-start]");
      if (!segment) {
        return;
      }
      event.preventDefault();
      audio.currentTime = Number(segment.dataset.start || 0);
      setActiveTranscriptSegment();
      audio.play();
    });
  }

  loadWaveform();
})();
