# SpeechPractice Smoke Test Checklist

Use this after changes that touch recording, transcription, history, playback, or export.

## Script Practice

- Start the app and confirm a script loads.
- Record a short take, stop recording, and confirm a new history row appears.
- Press Score and confirm the row shows `[scoring]`.
- While scoring is running, click at least one older history entry.
- Confirm the older entry remains readable and the background status says scoring is still running.
- When scoring finishes, confirm the original row returns to normal and the background status reports completion.
- Reopen the scored row and confirm transcript, metrics, highlights, waveform, and playback are available.

## Free Speak

- Enable Free Speak Mode.
- Record a short take and confirm transcription starts automatically.
- Click an older history row while transcription runs.
- Confirm the transcript worker keeps running and the background status updates on completion.
- Return to Free Speak and confirm Save is enabled after transcription.
- Save the Free Speak session and confirm it appears in history.

## Playback And Review

- Open a saved scripted session and play/pause audio.
- Click the waveform and confirm playback jumps.
- Confirm transcript highlighting follows the playhead when timestamped segments exist.
- Toggle Error Highlights and confirm script/transcript highlights hide and return.
- Use Copy Mistakes on a scored session with mistakes.

## Export

- Open a scored session and confirm Export is enabled.
- Export the report to `reports/`.
- Open the Markdown file and confirm it includes metrics, transcript, reference script, and mistakes.
- Open an untranscribed or blank session and confirm Export is disabled or reports no transcript.

## Quick Practice

- Enter Quick Practice mode.
- Paste a reference script, record a short take, and score it.
- Confirm the saved session is labeled Quick Practice and can be reopened from history.
