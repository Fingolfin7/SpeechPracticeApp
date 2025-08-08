from __future__ import annotations

from typing import List, Tuple

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QColor


def build_transcript_from_segments(segments_obj: object) -> tuple[str, list, list, int]:
    """
    Build transcript text and ranges from Whisper segments.
    Returns (text, segments_list, ranges, active_index)
    where ranges are tuples of (start_char, end_char, t0, t1).
    """
    segments: list = []
    ranges: List[Tuple[int, int, float, float]] = []
    active_index = -1
    text_parts: list[str] = []
    cursor_index = 0
    for seg in list(segments_obj):
        if not isinstance(seg, dict):
            continue
        seg_text = str(seg.get("text", ""))
        if seg_text == "":
            continue
        start_char = cursor_index
        text_parts.append(seg_text)
        cursor_index += len(seg_text)
        end_char = cursor_index
        try:
            t0 = float(seg.get("start", 0.0))
            t1 = float(seg.get("end", t0))
        except Exception:
            t0, t1 = 0.0, 0.0
        segments.append(seg)
        ranges.append((start_char, end_char, t0, t1))
    return "".join(text_parts), segments, ranges, active_index


def highlight_transcript_at_time(
    edit: QtWidgets.QTextEdit,
    ranges: List[Tuple[int, int, float, float]],
    t_seconds: float,
    active_index: int,
) -> int:
    """
    Highlight the segment covering time t_seconds; returns new active_index.
    """
    if not ranges:
        return -1
    idx = active_index
    if 0 <= idx < len(ranges):
        s_char, e_char, t0, t1 = ranges[idx]
        if t0 - 0.05 <= t_seconds <= t1 + 0.05:
            pass
        else:
            idx = -1
    if idx == -1:
        for i, (_s, _e, t0, t1) in enumerate(ranges):
            if t0 - 0.05 <= t_seconds <= t1 + 0.05:
                idx = i
                break
    if idx == -1 or idx == active_index:
        return active_index
    s_char, e_char, _t0, _t1 = ranges[idx]
    cursor = edit.textCursor()
    cursor.setPosition(max(0, int(s_char)))
    cursor.setPosition(max(0, int(e_char)), QtGui.QTextCursor.KeepAnchor)
    sel = QtWidgets.QTextEdit.ExtraSelection()
    sel.cursor = cursor
    fmt = QtGui.QTextCharFormat()
    fmt.setBackground(QColor("#3355ff55"))
    fmt.setProperty(QtGui.QTextFormat.FullWidthSelection, False)
    sel.format = fmt
    try:
        edit.setExtraSelections([sel])
        prev = edit.hasFocus()
        edit.setTextCursor(cursor)
        edit.ensureCursorVisible()
        if not prev:
            edit.clearFocus()
    except Exception:
        pass
    return idx


