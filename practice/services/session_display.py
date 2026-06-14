from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from django.utils.safestring import SafeString, mark_safe

from practice.models import PracticeSession, SessionError


@dataclass(frozen=True)
class HighlightedSessionText:
    script_html: SafeString
    transcript_html: SafeString
    error_count: int


def audio_exists(session: PracticeSession) -> bool:
    return bool(session.audio_path) and Path(session.audio_path).exists()


def session_segments(session: PracticeSession) -> list[dict]:
    if not session.segments:
        return []
    try:
        data = json.loads(session.segments)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def highlighted_session_text(session: PracticeSession) -> HighlightedSessionText:
    errors = list(SessionError.objects.filter(session_id=session.pk))
    script_ranges = []
    transcript_ranges = []
    for error in errors:
        if error.op in {"del", "sub"} and error.ref_start is not None and error.ref_end is not None:
            script_ranges.append((error.ref_start, error.ref_end, _script_class(error)))
        if error.op in {"ins", "sub"} and error.hyp_start is not None and error.hyp_end is not None:
            transcript_ranges.append((error.hyp_start, error.hyp_end, _transcript_class(error)))

    return HighlightedSessionText(
        script_html=_highlight_text(session.script_text or "", script_ranges),
        transcript_html=_highlight_text(session.transcript or "", transcript_ranges),
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
