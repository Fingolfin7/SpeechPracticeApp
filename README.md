# SpeechPractice

SpeechPractice is a Django app for scripted speech practice, free speaking, transcription, scoring, and progress review.

The current branch is the Django rebuild. The old Qt desktop GUI has been removed; use `manage.py` or `run_server.bat` to run the web app.

## Features

- Browser recording and audio upload.
- Scripted practice, quick practice, and free speak modes.
- Local Whisper, OpenAI transcription, or sidecar transcript providers.
- Async scoring jobs with live status and partial local-Whisper transcript updates.
- WER, CER, clarity score, articulation rate, pause ratio, confidence, and filled-pause metrics.
- Error highlights, clickable timestamped transcripts, waveform seeking, transcript edits, rescoring, and report export.
- Script library with manual scripts, imports, generated drills, ladders, tags, and spaced-repetition cards.
- Account settings for transcription/generation providers, API keys, Codex auth, local Whisper tuning, and Autumn timer metadata.

## Requirements

- Python 3.10-3.12 recommended, especially for CUDA/GPU acceleration.
- ffmpeg on `PATH` for Whisper and audio conversion.
- Optional NVIDIA CUDA-capable GPU plus a matching PyTorch build for faster local Whisper.

## Setup

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

For CUDA-enabled local Whisper, install the PyTorch wheel that matches your machine from the official PyTorch selector.

## Run

Double-click:

```text
run_server.bat
```

Or run manually:

```powershell
.\.venv312\Scripts\python.exe manage.py migrate
.\.venv312\Scripts\python.exe manage.py runserver 0.0.0.0:8000
```

Then open:

- Local: `http://127.0.0.1:8000/`
- LAN: `http://<your-computer-ip>:8000/`

## Useful Commands

```powershell
.\.venv312\Scripts\python.exe manage.py check
.\.venv312\Scripts\python.exe manage.py test
.\.venv312\Scripts\python.exe manage.py refresh_cards
.\.venv312\Scripts\python.exe manage.py process_scoring_jobs
```

## Data

Development data is intentionally local:

- `sessions.db`
- `recordings/`
- `reports/`
- `settings.json`
- `script_index.json`
- `scripts/`

These paths are ignored by git.

## Notes

- Local Whisper model instances are cached process-wide and can be cleared from the Account page after tuning changes.
- The sidecar transcript provider is useful for smoke tests: upload or point at an audio path with a sibling `.txt` transcript.
- See `SMOKE_TEST_CHECKLIST.md` for manual checks after touching recording, transcription, playback, or export flows.
