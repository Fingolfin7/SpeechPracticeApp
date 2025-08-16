---
title: Speech Practice App – Metric Targets and Rationale
tags:
  - speech
  - clarity
  - metrics
  - whisper
updated: 2025-08-16
---

# Speech Practice App – Metric Targets and Rationale

Purpose
- Practical targets for “clear” scripted English in quiet conditions
  (decent mic, Whisper base.en or better).
- Rationale for each metric so future-you remembers why these bands make sense.
- Pointers to where the code computes them.

## Quick Targets (reading, quiet room)

- WER (word error rate)
  - Excellent: ≤ 5%
  - Clear/Good: 5–10%
  - Work to do: > 10–20%
- CER (character error rate)
  - Excellent: ≤ 2%
  - Clear/Good: 2–4%
  - Work to do: > 4–8%
- Clarity (defined as 1 − WER)
  - Excellent: ≥ 95%
  - Clear/Good: 90–95%
  - Work to do: 80–90%
- Avg Confidence (Whisper avg_logprob normalized to 0–1)
  - Excellent: ≥ 0.85
  - Clear/Good: 0.75–0.85
  - Work to do: < 0.75
- Articulation Rate (words per minute; pauses excluded)
  - Clear reading: 120–160 wpm
  - Drill/precision practice: 100–120 wpm
  - If > ~170 wpm and ASR errors rise, slow down
- Pause Ratio (total pause time / total time)
  - Clear: 10–25%
  - Okay: 25–35%
  - Work to do: > 35% (choppy/hesitant) or < 8–10% (breathless/monotone)
- Filled pauses (um/uh) during scripted reading
  - Aim for < 2 per minute (reading; conversational speech will be higher)

## Why these numbers

- WER and CER
  - Whisper-family models achieve single-digit WER on clean, read English in
    zero‑shot settings; with smaller models and real rooms, 5–10% WER is a
    realistic “clear” target. CER is typically a few points lower than WER.
- Clarity
  - In this app clarity = 1 − WER, so it maps directly to those WER bands.
- Avg Confidence
  - Whisper emits `avg_logprob` per segment; this app maps roughly from
    [-1, 0] → [0, 1]. Higher session means correlate with clearer articulation
    and lower noise. Bands (0.75–0.85+) are pragmatic engineering targets;
    calibrate to your mic/room.
- Articulation Rate and Pauses
  - Clear reading lives around 120–160 wpm; articulation drills benefit from
    100–120 wpm. Voluntary rate reduction increases both pause time and reduces
    articulation rate; excessive pausing (> 35%) sounds hesitant; very low
    pausing (< ~10%) can sound breathless/monotone.
- Filled pauses
  - “Uh/um” signal upcoming delays. High frequency during reading often
    indicates planning/fluency issues; in spontaneous conversation they’re
    genre‑dependent and not inherently “bad.”

## How the app computes these

- Core ASR metrics (transcribe_worker.py)
  - WER: `jiwer.wer(reference, hypothesis)` after normalization
  - Clarity: `1.0 - WER`
  - Score (1–5): logistic map centered at clarity = 0.80

    ```python
    import math

    def scale_score(clarity: float) -> float:
        c = max(0.0, min(1.0, clarity))
        return max(1.0, min(5.0, 1 + 4 / (1 + math.exp(-20 * (c - 0.80)))))
    ```
- Extended metrics (transcription_service.py)
  - CER: `jiwer.cer` on the same normalized text used for WER
  - Avg Confidence: mean of normalized `avg_logprob` per segment
    (map ~[-1, 0] → [0, 1])
  - Articulation Rate (wpm): `words_in_hyp / speech_time_minutes`
    (speech_time excludes pauses)
  - Pause Ratio: `total_pause_time / total_time` (between first and last
    segment)
  - Filled pauses: count of tokens `{um, uh, erm, er, hmm}` in normalized hyp
- Visual error highlights (alignment_utils.py)
  - Word alignment (S/D/I) and character-level diffs; stronger red for deleted
    vowels to surface articulation targets. Legends are driven by a single
    palette in `highlight_theme.py`.

## Reading the combo (quick heuristics)

- High WER and high CER + low Conf → articulation/enunciation issues or noise;
  slow down and boost consonant energy, reduce noise.
- High WER but low CER + okay Conf → lexical or timing issues; work on pacing
  and phrasing.
- Good WER/CER but low Conf → mic/noise/room or “soft” delivery; fix the input
  chain and projection first.
- Rising Rate + rising WER/CER + falling Conf → outrunning clarity; target
  120–160 wpm and recheck.

## Calibration tips

- Calibrate to your setup: compute medians over the last 10–20 reading
  sessions (same mic/room) and track improvements vs your own baseline.
- Keep separate baselines for Reading vs Free Speak vs conversational drills.
- Optional: color metrics by quartiles relative to your history (top quartile =
  green, middle = amber, bottom = red).

## Caveats

- ASR-driven metrics are proxies, not ground truth; they’re most reliable on
  scripted English in quiet conditions.
- Confidence is model- and setup-dependent; recalibrate bands if you change
  mic/room/model.
- Pause ratio varies by genre and speaker; treat bands as coaching guides, not
  clinical cutoffs.

## Sources

- Whisper performance (targets for WER/CER/clarity)
  - Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak
    Supervision. arXiv. https://arxiv.org/abs/2212.04356
  - ICML 2023 version:
    https://proceedings.mlr.press/v202/radford23a.html
  - PDF: https://cdn.openai.com/papers/whisper.pdf
- Timing, rate reduction, and pauses
  - Tjaden, K., & Wilding, G. (2011). Speech and Pause Characteristics
    Associated with Voluntary Rate Reduction in Parkinson’s disease and
    Multiple Sclerosis. Journal of Communication Disorders, 44(6), 655–665.
    Open access (PMC):
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3202048/
- Filled pauses as signals
  - Clark, H. H., & Fox Tree, J. E. (2002). Using uh and um in spontaneous
    speaking. Cognition, 84, 73–111.
    https://www.sciencedirect.com/science/article/abs/pii/S0010027702000173
    (overview PDF mirror:
    http://www.columbia.edu/~rmk7/HC/HC_Readings/Clark_Fox.pdf)
  - Fox Tree, J. E. (2001). Listeners’ uses of um and uh in speech
    comprehension. (genre/context on listener inference)
