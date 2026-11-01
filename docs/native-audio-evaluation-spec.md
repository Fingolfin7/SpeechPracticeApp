# Experimental native-audio assessment

Status: **implemented as an isolated, undeployed MVP**. Add a deployment feature gate and decide who can access it **before merging or deploying**. No production change or validation is implied.

## Flow and boundaries

Authenticated users record or upload a take, listen to it, optionally supply a reference (empty by default), and explicitly request an assessment. The server sends the converted audio directly to `gpt-audio-1.5` through Chat Completions with text-only output. There is no Whisper call, transcript-derived metric, delegated model, automatic pairwise comparison or trend. Legacy sessions, errors, cards, ladders and scores are untouched. Experimental 1–5 ratings reflect unvalidated model judgments, not clinical measurements or direct human listener judgments.

The original recording and the exact 24 kHz mono PCM WAV sent to OpenAI are stored separately and deleted along with assessments when the user deletes the take. Recordings are limited to 10 MB and 120 seconds; ffmpeg must be installed. Conversion and provider failures retain the recording. Explicit retry creates a new assessment and can incur another charge. A unique submission ID and one-active-assessment constraint guard duplicate requests. SDK automatic retries are disabled. An attempt still running after five minutes may be explicitly marked interrupted; it is never automatically resent, and a provider call may have incurred charges even if the outcome is unknown.

This prototype performs assessment during the web request. It requires the Django worker to stay alive, and a browser, proxy or server timeout can interrupt the request before the provider responds. Check attempt status before retrying. Use durable background processing and review worker/proxy timeouts before deployment; a disconnected or killed worker does not prove the provider did not process the request.

The audio and optional reference go to OpenAI when the user assesses. `store=false` is requested but does not eliminate all provider retention: [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data) describe account-dependent retention and exceptions. Users own their recordings and may delete app-held copies; app deletion cannot promise deletion of provider-retained data. Access to takes, playback, assessment and deletion is authenticated and scoped to the owner.

The prompt instructs the model to listen to the audio and treat any reference as untrusted context, not evidence that a sound was clearly articulated. Structured outputs are not supported by this model. The application validates the response schema; refusals, abstentions, incomplete responses, provider errors and malformed output receive no structured ratings. Timestamps and accuracy are not guaranteed. Per attempt, it records the prompt version, full prompt including reference and hash, requested and returned model, request parameters, request ID and usage when returned, and the hash of the exact WAV placed in the request. A hash recorded before an error does not prove OpenAI received it. This supports auditability, not deterministic reproducibility or proof of model interpretation.

## Pilot: predeclared decisions

Before trying the model, record sample IDs and human listening judgments without seeing model results. Include repeated assessments of identical takes, a mix with and without a reference, deliberately clearer, faster and less clearly articulated variants, and degraded/background-noise samples. Where feasible add a second independent listener. Inspect every cited issue against playback, including possible accent-related penalties; record latency, errors, usage and costs. Do not claim validation from this small pilot.

- **Continue to a larger, independently reviewed pilot only if:** the model commonly identifies points human listeners can verify, suggestions are useful, repeat judgments are reasonably stable for the intended use, references do not systematically turn unclear speech into confident claims, degraded audio prompts appropriate uncertainty, and cost and latency are acceptable to the user.
- **Stop or redesign if:** it frequently invents problems, confidently rates unintelligible or degraded audio, penalizes accent independently of understandability, changes conclusions materially on repeats or with the reference, fails to abstain when needed, or costs/latency are unacceptable. Resolve privacy, reliability and feature-gating issues before any rollout.

These hypotheses are qualitative decision criteria, not pass rates or validation thresholds. Agree on sample counts and measurable thresholds before collecting a pilot dataset if a formal comparison is desired.

API interface: [Audio in Chat Completions](https://developers.openai.com/api/docs/guides/audio-chat-completions), [GPT-Audio-1.5](https://developers.openai.com/api/docs/models/gpt-audio-1.5).
