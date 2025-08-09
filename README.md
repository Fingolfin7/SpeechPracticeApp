# SpeechPracticeApp

<img width="1107" height="671" alt="image" src="https://github.com/user-attachments/assets/7fad2419-75f6-4fd8-8974-a231070e4312" />

SpeechPracticeApp is a Python application I built to support my speaking drills—helping me practice speaking clearly and track progress objectively. It records practice sessions, transcribes them with Whisper, and scores the result so I can target what to improve.

## Why this project?

I wanted an impartial, repeatable way to judge how clearly I speak. Human judgments are subjective and vary from day to day. By using a fixed “source script,” recording my practice, and transcribing what I actually said, I can compare the two texts and quantify how close my speech was to the target. This provides a consistent baseline for drills.

## How it works

1. Provide scripts (text prompts) as .txt files in the project’s Scripts/ directory (see “Scripts directory” below).
2. Start a session: the app will randomly select a script from Scripts/ (or you can pick one manually in Settings).
3. Record yourself speaking or upload an audio file.
4. The app uses Whisper to transcribe your speech.
5. It compares the transcription with the target text and reports:
   - Scores (overall/composite)
   - WER (Word Error Rate)
   - Clarity (experimental)

You can tune accuracy/speed via device (GPU vs CPU) and beam search settings in the Settings page.

## Scripts directory

- Location: a folder named Scripts/ at the project root (case-sensitive on Linux/macOS).
- Contents: one or more plain-text .txt files; each file is a “script” (one prompt per file).
- Encoding: UTF-8 recommended.
- Selection behavior: by default the app randomly chooses a script at session start; you can override and select a specific file in Settings.

Example structure:
```
SpeechPracticeApp/
├── Scripts/
│   ├── articulation_drill_1.txt
│   ├── tongue_twisters.txt
│   └── reading_passage_short.txt
└── ...
```

## Whisper transcriptions

Whisper is a state-of-the-art speech-to-text model. In this app, Whisper:
- Transcribes recorded audio to text.
- Provides timestamps and log-probability signals that can inform clarity-style metrics.
- Can run on CPU or GPU.

### GPU (CUDA) vs CPU

- GPU (CUDA): Much faster and recommended for interactive practice and longer audio. Requires a CUDA-capable NVIDIA GPU and a PyTorch build with CUDA.
- CPU: Works on any machine but is slower, especially with larger models.

Important Python version note for CUDA:
- Use Python <= 3.12 if you want GPU/CUDA acceleration. Python 3.13+ does not yet have CUDA-supported builds for this stack.
- CPU-only usage can work on newer Python versions, but for maximum compatibility (and to switch to GPU later), prefer Python 3.10–3.12.

## Scores | WER | Clarity

- Scores (overall): A single, user-facing number designed to summarize performance. It can combine multiple signals (e.g., WER, speaking-rate stability, pronunciation confidence) into one score. The exact combination can evolve over time.
- WER (Word Error Rate): A standard ASR metric computed by aligning the reference (target script) and hypothesis (Whisper transcription).
  - WER = (Substitutions + Deletions + Insertions) / Number of reference words.
  - Lower is better. It reflects intelligibility and correctness at the word level.
- Clarity (experimental): “Clarity” is hard to measure reliably because it blends pronunciation, enunciation, prosody, and noise. In this app, clarity is treated as an evolving metric. Potential ingredients include:
  - ASR-based signals: per-word confidence/log-probability, insertion/deletion-heavy errors (often correlate with mumbling), and timing stability.
  - Timing/fluency: articulation rate, pause ratio, filled pauses.
  - Audio quality proxies: SNR-like features, clipping detection.
  - Optional character-level error rate (CER) to catch subtle articulation issues missed by WER.


## Settings page

<img width="1104" height="673" alt="image" src="https://github.com/user-attachments/assets/71ba7162-f2e9-49a3-a213-ebde9c46bb83" />


Use the Settings page to:
- Select device: CPU or GPU (CUDA).
- Adjust beam size for decoding.
- Choose language/translation behavior (if applicable).
- Pick how the script is chosen: random from Scripts/ or a specific file.
- Configure audio input preferences.

## Beam settings

Whisper supports beam search decoding:
- Beam size: Higher values explore more candidate transcriptions, often improving accuracy at the cost of speed and memory.
- Tips:
  - CPU: Keep beam size modest (e.g., 1–3) for responsiveness.
  - GPU: You can raise beam size (e.g., 5–10) for better accuracy if latency is acceptable.


These controls help tailor accuracy, speed, and the feedback style to your goals.

## Getting started

Prerequisites:
- Python 3.10–3.12 recommended (Python <= 3.12 is required for CUDA/GPU acceleration; Python 3.13+ currently lacks CUDA support for this stack).
- ffmpeg installed and on your PATH.
- For GPU: NVIDIA drivers + a PyTorch build with CUDA.

Setup (example):
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install project dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch (pick the command for your CUDA version from pytorch.org)
# Example (replace cu121 with your CUDA version as needed):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# (Optional) CPU-only PyTorch:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Add your scripts:
```bash
mkdir -p Scripts
# Add one or more UTF-8 .txt files into the Scripts/ folder
# e.g., Scripts/articulation_drill_1.txt
```

Run the app:
```bash
# Example entry point; adjust to your repo’s actual start command
python main.py
# or
python -m speech_practice_app
```

## Interpreting results

- Use WER to track intelligibility and correctness against your target script.
- Watch the clarity score trend over time, but treat it as supplemental and evolving.
- Inspect insertions/deletions specifically: many deletions can indicate swallowing words; many insertions can reflect unclear boundaries or disfluencies.

<img width="1107" height="673" alt="image" src="https://github.com/user-attachments/assets/6184b28e-a204-4404-b8cd-a3b046044163" />


## Troubleshooting

- Torch says “not compiled with CUDA”: Ensure Python <= 3.12, and install a CUDA-enabled Torch wheel that matches your driver/CUDA.
- Whisper is slow: Use a smaller model, reduce beam size, switch to GPU.
- Transcripts drift off-script: Increase beam size slightly, set the correct language, reduce background noise, or re-record closer to the mic.

## Contributing

Issues and pull requests are welcome—especially improvements to clarity scoring, new metrics, or calibration methods.

## License

MIT License (see LICENSE file).
