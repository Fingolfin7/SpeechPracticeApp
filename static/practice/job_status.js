(function () {
  const panel = document.querySelector("[data-job-status-url]");
  if (!panel) {
    return;
  }

  const statusUrl = panel.dataset.jobStatusUrl;
  const statusLabel = document.querySelector("[data-job-status-label]");
  const liveState = document.querySelector("[data-job-live-state]");
  const transcriptBox = document.querySelector("[data-job-partial-transcript]");
  const nextAction = document.querySelector("[data-job-next-action]");
  let stopped = false;

  function render(payload) {
    if (statusLabel) {
      statusLabel.textContent = payload.status_label || payload.status;
    }
    if (transcriptBox && payload.partial_transcript) {
      transcriptBox.textContent = payload.partial_transcript;
    }
    if (liveState) {
      if (payload.is_pending) {
        liveState.textContent = payload.partial_transcript ? "Transcribing" : "Waiting";
      } else if (payload.is_failed) {
        liveState.textContent = "Failed";
      } else {
        liveState.textContent = "Finished";
      }
    }
    if (!payload.is_pending && nextAction) {
      nextAction.replaceChildren();
      if (payload.is_done && payload.session_url) {
        const message = document.createElement("p");
        message.textContent = "Scoring finished.";
        const link = document.createElement("a");
        link.className = "button-link";
        link.href = payload.session_url;
        link.textContent = "Open scored session";
        nextAction.append(message, link);
      } else if (payload.is_failed) {
        const message = document.createElement("p");
        message.className = "form-error";
        message.textContent = payload.error_message || "Scoring failed.";
        nextAction.appendChild(message);
      }
    }
    stopped = !payload.is_pending;
  }

  async function poll() {
    if (stopped) {
      return;
    }
    try {
      const response = await fetch(statusUrl, {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error("Job status failed.");
      }
      render(await response.json());
    } catch (error) {
      if (liveState) {
        liveState.textContent = "Status unavailable";
      }
    }
    if (!stopped) {
      window.setTimeout(poll, 1800);
    }
  }

  poll();
})();
