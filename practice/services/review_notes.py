from __future__ import annotations

import re


LEGACY_REVIEW_NOTES_RE = re.compile(
    r"Quality (\d+(?:\.\d+)?); mastery (\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?)"
)


def parse_legacy_review_notes(notes: str) -> tuple[float, float] | None:
    match = LEGACY_REVIEW_NOTES_RE.search(notes or "")
    if not match:
        return None
    return float(match.group(1)), float(match.group(3))
