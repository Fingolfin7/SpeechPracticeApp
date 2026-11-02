# SpeechPractice

SpeechPractice is a small local experiment for listening to your speech. Record up to two minutes, play it back, and ask GPT Audio 1.5 for practical feedback on clarity, pronunciation and mumbling. Feedback is experimental and may be mistaken; it is not a diagnosis.

## Start on Windows

Double-click `run_server.bat`, then open [http://127.0.0.1:8765/](http://127.0.0.1:8765/) in a browser. Keep the terminal window open while using the app. The launcher checks for dependencies and installs the small set in `requirements-studio.txt` if needed. It needs Python 3.12 and `ffmpeg` on PATH.

Record and stop, optionally listen back, enter an OpenAI API key, and choose **Send for feedback**. The key is kept in the page and sent only with that request; the app does not save or log it. The browser asks for microphone permission. Localhost is required; the app listens only on this computer.

The recording is sent directly to `gpt-audio-1.5` through Chat Completions with text-only output. No Whisper transcription, transcript scoring or separate model is involved. Each deliberate request may incur API charges. If a connection is interrupted, OpenAI may still have processed the request; check before sending again.

This app has no accounts, database, recording history or persistent recording storage. Audio conversion uses temporary files, removed after processing under normal operation; an abrupt process or computer crash could leave temporary files in the system temp directory. Closing or refreshing the page loses the current recording and feedback. OpenAI processes the recording and its [data retention policies](https://developers.openai.com/api/docs/guides/your-data) apply. `store=false` does not eliminate all provider retention.

The previous SpeechPractice product code remains in the repository for reference but is not started by `run_server.bat`. The standalone app uses `voice_studio` and does not access the older app's database, recordings, settings or scores.

## Development

```powershell
.\.venv312\Scripts\python.exe manage_studio.py check
.\.venv312\Scripts\python.exe manage_studio.py test voice_studio
.\.venv312\Scripts\python.exe manage_studio.py runserver 127.0.0.1:8765 --noreload --insecure
```

`--insecure` lets Django's development server serve this app's static assets while `DEBUG` is off. This loopback-only development server is intended for local use, not deployment or network access. The API call has a 90-second timeout and no automatic retries; feedback requires the server window to remain open.
