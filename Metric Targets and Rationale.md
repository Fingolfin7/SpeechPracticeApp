---
title: SpeechPractice - Metric Targets and Rationale
tags:
  - speech
  - clarity
  - metrics
  - whisper
updated: 2026-06-18
---

# SpeechPractice - Metric Targets and Rationale

Purpose:

- Practical targets for clear scripted English in quiet conditions.
- Rationale for each metric so the bands remain understandable.
- Pointers to where the Django app computes them.

## Quick Targets

- WER: excellent <= 5%; clear/good 5-10%; work to do > 10-20%.
- CER: excellent <= 2%; clear/good 2-4%; work to do > 4-8%.
- Clarity: excellent >= 95%; clear/good 90-95%; work to do 80-90%.
- Avg confidence: excellent >= 0.85; clear/good 0.75-0.85; work to do < 0.75.
- Articulation rate: clear reading 120-160 wpm; precision drills 100-120 wpm.
- Pause ratio: clear 10-25%; okay 25-35%; work to do > 35% or < 8-10%.
- Filled pauses during scripted reading: aim for fewer than 2 per minute.

## Why These Numbers

- WER and CER are practical intelligibility proxies for clean read speech.
- Clarity is currently `1 - WER`, so it maps directly to the WER bands.
- Whisper `avg_logprob` is normalized roughly from `[-1, 0]` to `[0, 1]`.
- Clear reading often lives around 120-160 wpm, while articulation drills benefit from a slower 100-120 wpm pace.
- Pause and filled-pause metrics are coaching signals, not clinical measurements.

## Current Code Pointers

- `practice/services/scoring.py`: WER, CER, clarity, score, articulation rate, pause ratio, filled pauses, confidence.
- `practice/services/transcription.py`: provider selection, local Whisper/OpenAI/sidecar transcription.
- `practice/services/local_whisper.py`: chunked local-Whisper transcription for long audio.
- `alignment_utils.py`: word alignment, flexible WER, and character-level diffs.
- `error_analytics.py`: persisted error events and trend summaries.
- `highlight_theme.py`: shared highlight palette.

## Quick Heuristics

- High WER and high CER plus low confidence: articulation, noise, or mic issues.
- High WER but low CER and decent confidence: pacing or phrasing issues.
- Good WER/CER but low confidence: input chain, room noise, or too-soft delivery.
- Rising rate with rising WER/CER and falling confidence: slow down and recheck.

## Calibration Tips

- Compare against your own recent median, not only the absolute bands.
- Keep separate baselines for scripted reading, drills, and free speak.
- Recalibrate after changing mic, room, model, provider, or language settings.

## Sources

- Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. https://arxiv.org/abs/2212.04356
- Tjaden, K., and Wilding, G. (2011). Speech and Pause Characteristics Associated with Voluntary Rate Reduction. Journal of Communication Disorders, 44(6), 655-665.
- Clark, H. H., and Fox Tree, J. E. (2002). Using uh and um in spontaneous speaking. Cognition, 84, 73-111.
