from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from django.conf import settings
from django.utils import timezone
from jiwer import cer as jiwer_cer

from alignment_utils import compute_flexible_wer
from error_analytics import extract_error_events

from practice.models import PracticeSession, SessionError
from practice.services.transcription import TranscriptResult, get_transcription_provider


@dataclass(frozen=True)
class ScoreResult:
    transcript: str
    wer: float
    cer: float
    clarity: float
    score: float
    segments: list[dict[str, Any]]
    artic_rate: float
    pause_ratio: float
    filled_pauses: float
    avg_conf: float | None
    provider: str


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s]", "", text)


def scale_score(clarity: float) -> float:
    clarity = max(0.0, min(clarity, 1.0))
    score = 1 + 4 / (1 + math.exp(-20 * (clarity - 0.80)))
    return min(5.0, max(1.0, score))


def norm_conf_from_logprob(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    return max(0.0, min(1.0, float(logprob) + 1.0))


def augment_segments_and_fluency(
    segments: list[dict[str, Any]],
    transcript: str,
) -> tuple[list[dict[str, Any]], float, float, float, float | None]:
    augmented: list[dict[str, Any]] = []
    prev_end: float | None = None
    speech_time = 0.0
    pause_time = 0.0
    confidences: list[float] = []

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        duration = max(0.0, end - start)
        pause_before = 0.0
        if prev_end is not None:
            pause_before = max(0.0, start - prev_end)
            pause_time += pause_before
        prev_end = end
        speech_time += duration

        confidence = norm_conf_from_logprob(segment.get("avg_logprob"))
        if confidence is not None:
            confidences.append(confidence)

        copy = dict(segment)
        copy["duration"] = duration
        copy["pause_before"] = pause_before
        if confidence is not None:
            copy["conf"] = confidence
        augmented.append(copy)

    total_time = 0.0
    if segments:
        total_time = max(
            0.0,
            float(segments[-1].get("end", 0.0)) - float(segments[0].get("start", 0.0)),
        )

    word_count = len(clean_text(transcript).split())
    artic_rate = float(word_count) * (60.0 / speech_time) if speech_time > 1e-6 else 0.0
    denom = total_time if total_time > 1e-6 else speech_time + pause_time
    pause_ratio = pause_time / denom if denom > 1e-6 else 0.0
    filled_pauses = float(
        sum(1 for token in clean_text(transcript).split() if token in {"um", "uh", "erm", "er", "hmm"})
    )
    avg_conf = float(sum(confidences) / len(confidences)) if confidences else None
    return augmented, artic_rate, pause_ratio, filled_pauses, avg_conf


def score_transcript(script_text: str, transcript_result: TranscriptResult) -> ScoreResult:
    ref = clean_text(script_text)
    hyp = clean_text(transcript_result.text)
    wer = compute_flexible_wer(ref, hyp)
    clarity = max(0.0, 1.0 - wer)
    score = scale_score(clarity)
    cer_value = float(jiwer_cer(ref.replace(" ", ""), hyp.replace(" ", ""))) if ref or hyp else 0.0
    segments, artic_rate, pause_ratio, filled_pauses, avg_conf = augment_segments_and_fluency(
        transcript_result.segments,
        hyp,
    )
    return ScoreResult(
        transcript=hyp,
        wer=wer,
        cer=cer_value,
        clarity=clarity,
        score=score,
        segments=segments,
        artic_rate=artic_rate,
        pause_ratio=pause_ratio,
        filled_pauses=filled_pauses,
        avg_conf=avg_conf,
        provider=transcript_result.provider,
    )


def transcribe_score_and_store(
    *,
    user,
    script_name: str,
    script_text: str,
    audio_path: str,
    stored_audio_ref: str | None = None,
    provider_name: str | None = None,
    partial_callback: Callable[[str], None] | None = None,
) -> PracticeSession:
    provider = get_transcription_provider(provider_name, user=user)
    transcript = provider.transcribe(audio_path, partial_callback=partial_callback)
    result = score_transcript(script_text, transcript)
    session = PracticeSession.objects.create(
        user=user,
        timestamp=timezone.localtime().strftime("%Y-%m-%dT%H:%M:%S"),
        script_name=script_name,
        script_text=script_text,
        audio_path=stored_audio_ref or str(Path(audio_path).resolve()),
        transcript=result.transcript,
        wer=result.wer,
        clarity=result.clarity,
        score=result.score,
        segments=json.dumps(result.segments) if result.segments else None,
        cer=result.cer,
        artic_rate=result.artic_rate,
        pause_ratio=result.pause_ratio,
        filled_pauses=result.filled_pauses,
        avg_conf=result.avg_conf,
    )
    _replace_session_errors(session)
    return session


def transcribe_free_and_store(
    *,
    user,
    audio_path: str,
    stored_audio_ref: str | None = None,
    provider_name: str | None = None,
    partial_callback: Callable[[str], None] | None = None,
) -> PracticeSession:
    provider = get_transcription_provider(provider_name, user=user)
    transcript = provider.transcribe(audio_path, partial_callback=partial_callback)
    segments, artic_rate, pause_ratio, filled_pauses, avg_conf = augment_segments_and_fluency(
        transcript.segments,
        transcript.text,
    )
    return PracticeSession.objects.create(
        user=user,
        timestamp=timezone.localtime().strftime("%Y-%m-%dT%H:%M:%S"),
        script_name="Free Speak",
        script_text="",
        audio_path=stored_audio_ref or str(Path(audio_path).resolve()),
        transcript=transcript.text.strip(),
        wer=None,
        clarity=None,
        score=None,
        segments=json.dumps(segments) if segments else None,
        cer=None,
        artic_rate=artic_rate,
        pause_ratio=pause_ratio,
        filled_pauses=filled_pauses,
        avg_conf=avg_conf,
    )


def _replace_session_errors(session: PracticeSession) -> None:
    SessionError.objects.filter(user=session.user, session_id=session.id).delete()
    events = extract_error_events(session.script_text, session.transcript or "")
    rows = [
            SessionError(
                user=session.user,
                session_id=session.id,
            timestamp=session.timestamp,
            script_name=session.script_name,
            ref_token=event.get("ref_token"),
            hyp_token=event.get("hyp_token"),
            op=event.get("op", ""),
            error_kind=event.get("error_kind", ""),
            ref_start=event.get("ref_start"),
            ref_end=event.get("ref_end"),
            hyp_start=event.get("hyp_start"),
            hyp_end=event.get("hyp_end"),
            ref_local_start=event.get("ref_local_start"),
            ref_local_end=event.get("ref_local_end"),
            hyp_local_start=event.get("hyp_local_start"),
            hyp_local_end=event.get("hyp_local_end"),
            ref_token_len=event.get("ref_token_len"),
            hyp_token_len=event.get("hyp_token_len"),
            confidence=event.get("confidence"),
            segment_start=event.get("segment_start"),
            segment_end=event.get("segment_end"),
        )
        for event in events
    ]
    if rows:
        SessionError.objects.bulk_create(rows)
