(function () {
  const copyScoreButton = document.querySelector("[data-copy-score]");
  const scoreCopyText = document.querySelector("[data-score-copy-text]");
  const copyMistakesButton = document.querySelector("[data-copy-mistakes]");
  const mistakeLines = document.querySelector("[data-mistake-lines]");
  const toggleHighlightsButton = document.querySelector("[data-toggle-highlights]");
  const compareGrid = document.querySelector(".compare-grid");

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

  function flashButtonText(button, text, restoreText) {
    button.textContent = text;
    window.setTimeout(function () {
      button.textContent = restoreText;
    }, 1300);
  }

  if (copyScoreButton && scoreCopyText) {
    copyScoreButton.addEventListener("click", async function () {
      const text = scoreCopyText.textContent.trim();
      if (!text) {
        return;
      }
      try {
        const copied = await copyTextToClipboard(text);
        if (copied) {
          flashButtonText(copyScoreButton, "Copied", "Copy score");
        }
      } catch (error) {
        const range = document.createRange();
        range.selectNodeContents(scoreCopyText);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
      }
    });
  }

  if (copyMistakesButton && mistakeLines) {
    copyMistakesButton.addEventListener("click", async function () {
      const text = mistakeLines.textContent.trim();
      if (!text) {
        return;
      }
      try {
        const copied = await copyTextToClipboard(text);
        if (copied) {
          flashButtonText(copyMistakesButton, "Copied", "Copy mistakes");
        }
      } catch (error) {
        const range = document.createRange();
        range.selectNodeContents(mistakeLines);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
      }
    });
  }

  if (toggleHighlightsButton && compareGrid) {
    toggleHighlightsButton.addEventListener("click", function () {
      compareGrid.classList.toggle("hide-error-highlights");
    });
  }

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
    const segments = Array.from(timedTranscript.querySelectorAll("[data-start][data-end]"));
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

  function closestTimedSegment(target) {
    if (!target) {
      return null;
    }
    const element = target instanceof Element ? target : target.parentElement;
    return element ? element.closest("[data-start]") : null;
  }

  function seekAudioTo(targetTime) {
    const safeTime = Math.max(0, Number(targetTime) || 0);
    function applySeek() {
      audio.currentTime = safeTime;
      setActiveTranscriptSegment();
      const playPromise = audio.play();
      if (playPromise) {
        playPromise.catch(function () {
          if (canvas && waveformBuffer) {
            drawWaveform(waveformBuffer);
            drawPlayhead();
          }
        });
      }
    }

    if (audio.readyState < 1) {
      audio.addEventListener("loadedmetadata", applySeek, { once: true });
      audio.load();
      return;
    }
    applySeek();
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
    if (audioContext && audioContext.state === "suspended") {
      audioContext.resume();
    }
    if (waveformBuffer) {
      drawWaveform(waveformBuffer);
      drawPlayhead();
    }
    canvas.dataset.playState = "requested";
    seekAudioTo(targetTime);
    window.setTimeout(function () {
      if (!audio.paused) {
        canvas.dataset.playState = "playing";
      } else {
        canvas.dataset.playState = audio.paused ? "paused-after-play" : "playing";
      }
    }, 50);
  }

  canvas.addEventListener("pointerdown", seekAndPlay);
  canvas.addEventListener("click", seekAndPlay);
  audio.addEventListener("timeupdate", function () {
    if (waveformBuffer) {
      drawWaveform(waveformBuffer);
      drawPlayhead();
    }
    setActiveTranscriptSegment();
  });
  audio.addEventListener("loadedmetadata", setActiveTranscriptSegment);

  if (timedTranscript) {
    timedTranscript.addEventListener("click", function (event) {
      const segment = closestTimedSegment(event.target);
      if (!segment) {
        return;
      }
      seekAudioTo(segment.dataset.start);
    });
    timedTranscript.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      const segment = closestTimedSegment(event.target);
      if (!segment) {
        return;
      }
      event.preventDefault();
      seekAudioTo(segment.dataset.start);
    });
  }

  loadWaveform();
})();
