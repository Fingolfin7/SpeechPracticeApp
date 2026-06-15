# Django Migration Feature Completeness Plan

## Current Django Coverage

- Dashboard with due improvement cards, recent scoring jobs, recent scored sessions, coach notes, and error-pattern summaries.
- Browser-based recording/upload, waveform preview, async scoring jobs, retry, and history persistence into the legacy `sessions` table.
- Script library with create/edit/delete, JSON/CSV/text/zip import, generated drills, tags, source metadata, and random script selection in the practice UI.
- Improvement cards with basic spaced repetition, generated drills, review updates from scored sessions, and refresh from historical errors.
- Session detail with replay audio, waveform seeking, error highlights, transcript editing, rescore, transcript clearing, and recording deletion.
- Account settings for provider selection, OpenAI/Anthropic models, encrypted API keys, Codex login, and Autumn token storage.

## Missing Or Partial Original Features

- **Timestamped transcript sync:** Original transcript segments highlighted during playback, and clicking transcript text sought the audio. Django only had waveform seek.
- **Mode parity:** Original app had Script, Quick Practice, and Free Speak. Django only had one scripted scoring flow.
- **Whisper/model tuning:** Original settings included local Whisper model, device, preset, beam size, temperature, timestamps, no-speech threshold, conditioning, language, chunk length, and hardware guidance. Django had only provider/model/API settings.
- **Progress tracker:** Original had a dedicated progress window with date range, script filter, score/WER/clarity charts, and word/character/position/phoneme trend summaries. Django only showed static dashboard summaries.
- **Exports and mistake utilities:** Original could export the current report and copy mistakes. Django had no session report export and no mistake-only output.
- **Autumn timer depth:** Original tracked Autumn project/subproject selection and active timer state. Django stored token/base URL only.
- **Live partial transcription:** Original showed partial chunk output while long transcription ran. Django queues jobs and refreshes status only.
- **Error highlight toggles:** Original could toggle highlights on/off. Django always renders highlights.
- **Model lifecycle controls:** Original unloaded/reloaded local Whisper after settings changes. Django caches models process-wide and does not yet expose a UI action for cache clearing.

## Porting Strategy

1. Restore playback parity by making session transcripts clickable and time-synced when segment timestamps exist.
2. Expand practice modes with Scripted, Quick Practice, and Free Speak while preserving the existing scoring job pipeline.
3. Move original Whisper tuning fields into `PracticeSettings` and have the Django transcription provider use them.
4. Add a dedicated Progress page with date/script filters, canvas charts, and the original trend summaries.
5. Add lightweight session report export and mistake-copy data to history pages.
6. Leave deeper live partial transcription and Autumn active timer controls as follow-up work after the web worker/job architecture is settled.

## Implementation Status

- [x] Audit current Django branch and original desktop feature surfaces.
- [x] Add synced/clickable timed transcript on session detail.
- [x] Add recording mode support, including Free Speak transcription-only sessions.
- [x] Expand account settings with local Whisper tuning and Autumn project metadata.
- [x] Add dedicated progress tracker page with charts and filters.
- [x] Add session report export and mistake-copy support.
- [x] Re-run tests and update this plan with remaining gaps.
- [x] Add live job-status polling with partial transcript persistence for chunked local transcription.
- [x] Add Autumn timer start/stop controls backed by saved project/subproject settings.
- [x] Add explicit local Whisper cache clearing after tuning changes and from Account.

## Remaining Gaps After Current Port

- OpenAI transcription still reports only the final transcript because the upstream request is not chunk-streamed in this app.
- Local live partials are available for long recordings through the existing chunked Whisper helper and job-status polling; short local recordings may only publish a final transcript because Whisper processes them in one call.
