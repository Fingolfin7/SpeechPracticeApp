(function () {
  const form = document.querySelector("[data-recorder-form]");
  if (!form) {
    return;
  }

  const startButton = form.querySelector("[data-record-start]");
  const stopButton = form.querySelector("[data-record-stop]");
  const status = form.querySelector("[data-recording-status]");
  const fileInput = form.querySelector("input[type='file']");
  const preview = form.querySelector("[data-recording-preview]");
  const stage = form.querySelector("[data-recording-stage]");
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

  if (!navigator.mediaDevices || !window.MediaRecorder) {
    if (startButton && status) {
      startButton.disabled = true;
      status.textContent = "Browser recording unavailable. Upload an audio file instead.";
    }
    return;
  }

  function setRecording(active) {
    startButton.disabled = active;
    stopButton.disabled = !active;
    stage.classList.toggle("is-recording", active);
    status.textContent = active ? "Recording..." : "Recording ready";
  }

  startButton.addEventListener("click", async function () {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
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
        preview.src = URL.createObjectURL(blob);
        preview.hidden = false;
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
      });
      recorder.start();
      setRecording(true);
    } catch (error) {
      status.textContent = "Microphone access failed. Use file upload instead.";
    }
  });

  stopButton.addEventListener("click", function () {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  });
})();
