# highlight_theme.py
from __future__ import annotations

from typing import Dict


# Central color palette (semi-transparent chips so text remains readable)
# Update here to change colors app-wide (alignment, legends, playhead).
SCRIPT_WORD_MISSING = "#ff6b6b55"  # whole missing word in script
REF_VOWEL_DELETE = "#ff3b3066"  # deleted vowel characters (script)
REF_CONS_DELETE = "#ff9e9e66"  # deleted non-vowel characters (script)

TRANSCRIPT_WORD_INSERT = "#6b9bff55"  # whole inserted word in transcript
TRANSCRIPT_CHAR_INSERT = "#66a3ff66"  # inserted characters (transcript)

REPLACE = "#ffc10766"  # character ranges replaced (both sides)

PLAYHEAD = "#3355ff55"  # transcript time-cursor overlay


def palette() -> Dict[str, str]:
    return {
        "script.word_missing": SCRIPT_WORD_MISSING,
        "script.vowel_delete": REF_VOWEL_DELETE,
        "script.cons_delete": REF_CONS_DELETE,
        "transcript.word_insert": TRANSCRIPT_WORD_INSERT,
        "transcript.char_insert": TRANSCRIPT_CHAR_INSERT,
        "both.replace": REPLACE,
        "playhead": PLAYHEAD,
    }


def _chip(label: str, color: str) -> str:
    return (
        f'<span style="background:{color}; padding:2px 6px; '
        f'border-radius:3px;">{label}</span>'
    )


def legend_html_for_script() -> str:
    p = palette()
    return "Legend: " + " ".join(
        [
            _chip("missing word", p["script.word_missing"]),
            _chip("deleted vowel", p["script.vowel_delete"]),
            _chip("deleted letter", p["script.cons_delete"]),
            _chip("replaced letters", p["both.replace"]),
        ]
    )


def legend_html_for_transcript() -> str:
    p = palette()
    return "Legend: " + " ".join(
        [
            _chip("inserted word", p["transcript.word_insert"]),
            _chip("inserted letters", p["transcript.char_insert"]),
            _chip("replaced letters", p["both.replace"]),
            _chip("playhead", p["playhead"]),
        ]
    )
