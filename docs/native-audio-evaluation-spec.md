# Experimental spec: native-audio speech assessment

Status: **proposal / unimplemented**, on `experiment/native-audio-evaluation`. This branch is an isolated replacement experiment, **not** a Whisper add-on to the existing scoring pipeline. No production scoring change is implied by this document.

## Goal

Evaluate a speaker's **human-facing intelligibility and delivery from their recorded audio**, not whether a speech recognizer can recover a known script. Give the user a short rating, specific evidence of where delivery became difficult to follow, and actionable guidance. The primary user is practising clearer spoken English, particularly articulation and mumbling.

## Minimal user flow (first prototype)

1. Open a simple experimental recording page (existing browser microphone capture may be reused).
2. Optionally paste/select a reference passage. The reference is context for detecting where words may have stumbled; it must **not** cause the model to assume that unclear sounds were articulated clearly.
3. Press Record, speak, then press Stop. Show playback and let the user submit or discard the take.
4. Send **audio itself** plus a versioned standing evaluation prompt to an audio-native model. Do **not** first turn it into a Whisper transcript or calculate WER/CER/clarity from a transcript.
5. Show a compact report: overall intelligibility, articulation, pacing, specific audible stumbling points, and two concrete things to practise next.
6. Save the recording reference, model identifier, prompt version, optional passage, assessment, and any errors. Allow replay and comparison between takes.

For a first pass, use the API model `gpt-live-1` for a session-based proof of concept if its actual API flow supports the intended scoring response. **GPT-Live-1.1 is the ChatGPT product's voice-model name, not a separately documented API SKU**. Because Live is optimised for continuous conversation and may delegate reasoning, compare a simpler **recording-in, assessment-out** experiment with `gpt-audio-1.5` over Chat Completions before choosing an API implementation. Avoid claiming these models give clinically validated articulation scores.

## Draft evaluation prompt (version 0)

> You are a speech-intelligibility practice evaluator. Listen to the **actual audio** as a first-time English-speaking listener. Judge how easy the speaker is to understand, independently of how well an automatic transcription system might reconstruct the words. If a reference passage is provided, use it only to locate intended phrases, not to excuse unclear delivery. Give evidence-based, specific feedback about audible articulation, consonant endings, word boundaries, pacing, hesitations, and relevant pauses. Do not infer a problem you cannot hear; do not diagnose disorders or make claims about personality, native language, or identity. Different accents are acceptable; assess intelligibility rather than accent conformity. Flag uncertainty and avoid false precision. Give at most three concrete observations and two suggested exercises. Return the following fields in concise machine-readable text: overall_intelligibility_1_to_5, articulation_1_to_5, pacing_1_to_5, stumbling_points (each with an approximate location if verifiable, an audible observation, and confidence), strengths, next_steps, and uncertainty_notes.

Use stable anchors such as short quoted phrases where exact timestamps cannot be justified. If the chosen model has no guaranteed structured-output mode, validate and normalize the returned content in the application, and gracefully display a plain-language response on parse failure. **Do not fabricate word-level timings.**

## Proposed data contract

- **Inputs:** authorized user, audio recording (or stream), optional reference passage, requested evaluation prompt version.
- **Outputs:** model and prompt identifiers, three ordinal 1-to-5 judgments clearly labelled as *experimental*, an array of no more than three audible stumbling observations, two suggested next exercises, uncertainties, duration, provider usage / estimated cost if available.
- **Observations:** short excerpt or approximate interval, concrete symptom (for example: swallowed ending, compressed word boundary, rushed phrase, long hesitation), confidence: low/medium/high. The model should be allowed to return **no issue detected**.
- **Failure handling:** report unavailable/invalid assessment without giving a bogus perfect or zero score; preserve recording so user can retry deliberately; never log API keys.

## What this replaces and what it does not

- Replace the experiment's default **Whisper transcription -> reference comparison -> WER-derived clarity and score** path.
- Do not reuse `clarity = 1 - WER` or the existing logistic score as an intelligibility rating.
- Keep the existing `master` application unchanged while evaluating this alternative. Reusing browser recording, authentication, audio storage, playback, and session UI is fine. The experiment should not silently mutate historical records or break previous reports.
- A model's assertion that it detected a stumble is **not** ground truth. Do not imply clinical validation or a normative interpretation of 1–5 ratings.

## Acceptance / pilot evaluation

- A user can record, submit, receive and replay a model assessment without a Whisper call in that experiment's scoring path.
- Feedback identifies **audible** locations or phrases and actionable next steps, not generic praise or unsupported specificity.
- Run identical takes at least twice (to probe rating variability), then test intentionally clearer/slurred/fast variants and a few background-noise cases. Compare outputs with the user's own playback judgments; where practical, obtain a second human listener's judgment on a small sample.
- Check whether the model penalizes accent rather than difficulty of understanding. Check hallucinated mistakes against playback.
- Track **session duration**, total API usage, latency, and errors before defaulting to this method.
- Keep scores and trends separate from legacy Whisper metrics; the numbers have different meanings.

## Cost and model selection notes (checked 2026-11-01)

- `whisper-1` transcription: **$0.006 per audio minute**.
- `gpt-live-1`: **$0.05 per active Live session minute**, billed per second, **plus any delegated backend model/tool usage**. A ten-minute session costs about **$0.50** for the voice layer alone. Session elapsed time may differ from audio recording length, especially if the app leaves a session open while the user reviews results.
- This is approximately 8.33x the Whisper per-minute baseline **only when session duration equals recording duration** and excludes backend charges. It is not an all-inclusive evaluation quote.
- `gpt-audio-1.5` audio input is charged per audio token (published rate $32 / million input audio tokens), so benchmark actual usage instead of translating its price into a confident flat per-minute estimate.

References:
- https://developers.openai.com/api/docs/models/gpt-live-1
- https://developers.openai.com/api/docs/models/gpt-audio-1.5
- https://developers.openai.com/api/docs/models/whisper-1
- https://developers.openai.com/api/docs/guides/live-delegation
- https://developers.openai.com/api/docs/pricing

## Out of scope for the first experiment

No automatic rewrite of practice scripts, no phoneme-level diagnosis without dedicated validation, no forced pronunciation targets, no migration of historical WER records, no mandatory speech-generating conversation, and no production rollout decision before the pilot.
