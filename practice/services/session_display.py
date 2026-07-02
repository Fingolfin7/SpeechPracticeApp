from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Iterable

from django.utils.safestring import SafeString, mark_safe

from practice.models import PracticeSession, SessionError
from practice.services.audio_storage import audio_exists as stored_audio_exists


@dataclass(frozen=True)
class HighlightedSessionText:
    script_html: SafeString
    transcript_html: SafeString
    timed_transcript_html: SafeString
    has_timed_transcript: bool
    error_count: int


def audio_exists(session: PracticeSession) -> bool:
    return stored_audio_exists(session.audio_path)


def session_segments(session: PracticeSession) -> list[dict]:
    if not session.segments:
        return []
    try:
        data = json.loads(session.segments)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def highlighted_session_text(session: PracticeSession) -> HighlightedSessionText:
    errors = list(SessionError.objects.filter(user=session.user, session_id=session.pk))
    script_ranges = []
    transcript_ranges = []
    for error in errors:
        if error.op in {"del", "sub"} and error.ref_start is not None and error.ref_end is not None:
            script_ranges.append((error.ref_start, error.ref_end, _script_class(error)))
        if error.op in {"ins", "sub"} and error.hyp_start is not None and error.hyp_end is not None:
            transcript_ranges.append((error.hyp_start, error.hyp_end, _transcript_class(error)))

    transcript_text = session.transcript or ""
    return HighlightedSessionText(
        script_html=_highlight_text(session.script_text or "", script_ranges),
        transcript_html=_highlight_text(transcript_text, transcript_ranges),
        timed_transcript_html=_timed_transcript_html(session, transcript_ranges),
        has_timed_transcript=bool(_segment_ranges_for_transcript(transcript_text, session_segments(session))),
        error_count=len(errors),
    )


def _script_class(error: SessionError) -> str:
    if error.op == "del" or error.error_kind == "word_missing":
        return "err-missing"
    if error.error_kind == "vowel_delete":
        return "err-vowel"
    if error.error_kind == "cons_delete":
        return "err-consonant"
    return "err-replace"


def _transcript_class(error: SessionError) -> str:
    if error.op == "ins" or error.error_kind == "word_insert":
        return "err-insert"
    if error.error_kind == "char_insert":
        return "err-char-insert"
    return "err-replace"


def _highlight_text(text: str, ranges: Iterable[tuple[int, int, str]]) -> SafeString:
    if not text:
        return mark_safe("<span class=\"empty\">No transcript yet.</span>")
    normalized = _normalize_offsets(text)
    active_ranges = []
    for start, end, css_class in ranges:
        start_idx = normalized.get(max(0, int(start)))
        end_idx = normalized.get(max(0, int(end)))
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            continue
        active_ranges.append((start_idx, end_idx, css_class))
    active_ranges.sort(key=lambda item: (item[0], item[1]))

    pieces = []
    cursor = 0
    for start, end, css_class in active_ranges:
        if start < cursor:
            continue
        pieces.append(html.escape(text[cursor:start]))
        pieces.append(f'<mark class="error-chip {css_class}">{html.escape(text[start:end])}</mark>')
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return mark_safe("".join(pieces))


def _timed_transcript_html(
    session: PracticeSession,
    error_ranges: Iterable[tuple[int, int, str]],
) -> SafeString:
    text = session.transcript or ""
    ranges = _segment_ranges_for_transcript(text, session_segments(session))
    if not ranges:
        return _highlight_text(text, error_ranges)

    pieces = []
    cursor = 0
    error_ranges_list = list(error_ranges)
    for index, (start, end, start_time, end_time) in enumerate(ranges):
        if start > cursor:
            pieces.append(str(_highlight_text(text[cursor:start], _shift_ranges(error_ranges_list, cursor, start))))
        segment_html = _highlight_text(text[start:end], _shift_ranges(error_ranges_list, start, end))
        pieces.append(
            '<span class="timed-transcript-segment" '
            f'data-transcript-index="{index}" '
            f'data-start="{start_time:.3f}" '
            f'data-end="{end_time:.3f}" '
            'tabindex="0" role="button">'
            f"{segment_html}"
            "</span>"
        )
        cursor = end
    if cursor < len(text):
        pieces.append(str(_highlight_text(text[cursor:], _shift_ranges(error_ranges_list, cursor, len(text)))))
    return mark_safe("".join(pieces))


def _shift_ranges(
    ranges: Iterable[tuple[int, int, str]],
    start: int,
    end: int,
) -> list[tuple[int, int, str]]:
    shifted = []
    for range_start, range_end, css_class in ranges:
        overlap_start = max(start, int(range_start))
        overlap_end = min(end, int(range_end))
        if overlap_end > overlap_start:
            shifted.append((overlap_start - start, overlap_end - start, css_class))
    return shifted


def _segment_ranges_for_transcript(
    transcript_text: str,
    segments: list[dict],
) -> list[tuple[int, int, float, float]]:
    if not transcript_text or not segments:
        return []
    ranges = []
    normalized_text, offset_map = _normalized_text_with_offsets(transcript_text)
    cursor = 0
    for segment in segments:
        cleaned = _clean_segment_text(str(segment.get("text", "")))
        if not cleaned:
            continue
        found = normalized_text.find(cleaned, cursor)
        if found < 0:
            found = normalized_text.find(cleaned)
        if found < 0:
            continue
        clean_end = found + len(cleaned)
        start = offset_map.get(found)
        end = offset_map.get(clean_end)
        if start is None or end is None or end <= start:
            continue
        try:
            start_time = float(segment.get("start", 0.0))
            end_time = float(segment.get("end", start_time))
        except (TypeError, ValueError):
            continue
        ranges.append((start, end, start_time, end_time))
        cursor = end
    return ranges


def _clean_segment_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s]", "", text)


def _normalized_text_with_offsets(text: str) -> tuple[str, dict[int, int]]:
    """
    Build the same punctuation-light text used for segment matching while keeping
    a map back to display-text offsets. This lets timestamp spans survive when a
    transcript has punctuation or uneven whitespace.
    """
    parts: list[str] = []
    offsets: dict[int, int] = {0: 0}
    clean_idx = 0
    in_space = False
    for idx, char in enumerate(text.lower()):
        if char.isspace():
            if not in_space and parts:
                parts.append(" ")
                clean_idx += 1
                offsets[clean_idx] = idx + 1
            in_space = True
            continue
        if char.isalnum() or char == "_":
            parts.append(char)
            clean_idx += 1
            offsets[clean_idx] = idx + 1
            in_space = False
            continue
        in_space = False
    if parts and parts[-1] == " ":
        parts.pop()
        clean_idx -= 1
        offsets[clean_idx] = len(text)
    offsets[len(parts)] = len(text) if parts else 0
    return "".join(parts), offsets


def _normalize_offsets(text: str) -> dict[int, int]:
    offsets = {0: 0}
    clean_idx = 0
    in_space = False
    for idx, char in enumerate(text):
        if char.isspace():
            if not in_space:
                clean_idx += 1
                offsets[clean_idx] = idx + 1
            in_space = True
            continue
        in_space = False
        if char.isalnum() or char == "_":
            clean_idx += 1
            offsets[clean_idx] = idx + 1
    return offsets
