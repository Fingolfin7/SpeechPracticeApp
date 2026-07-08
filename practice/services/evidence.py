from __future__ import annotations

from collections.abc import Sequence

from error_analytics import (
    _phoneme_stats,
    _position_bucket,
    _position_stats,
    clean_text_for_alignment,
)

from practice.models import ImprovementCard, PracticeSession

SHRINKAGE_K = 3

ARTIC_RATE_IDEAL_MIN = 120.0
ARTIC_RATE_IDEAL_MAX = 160.0
ARTIC_RATE_DECAY = 40.0
PAUSE_RATIO_IDEAL_MIN = 0.10
PAUSE_RATIO_IDEAL_MAX = 0.25
PAUSE_RATIO_DECAY = 0.15
FILLED_PAUSES_DECAY = 6.0


def session_quality(session: PracticeSession) -> float:
    score = float(session.score or 0.0)
    wer = float(session.wer or 1.0)
    score_quality = max(0.0, min(1.0, (score - 1.0) / 4.0))
    wer_quality = max(0.0, min(1.0, 1.0 - wer))
    return round((score_quality * 0.65) + (wer_quality * 0.35), 3)


def card_evidence(card: ImprovementCard, session: PracticeSession, events: Sequence) -> dict | None:
    if card.kind == ImprovementCard.KIND_WORD:
        return _word_evidence(card.target_key, session, events)
    if card.kind == ImprovementCard.KIND_SOUND:
        return _sound_evidence(card.target_key, session, events)
    if card.kind == ImprovementCard.KIND_CHARACTER:
        return _character_evidence(card.target_key, session, events)
    if card.kind == ImprovementCard.KIND_POSITION:
        return _position_evidence(card.target_key, session, events)
    return None


def quality_for_card(
    card: ImprovementCard,
    session: PracticeSession,
    events: Sequence,
) -> float | None:
    quality, _evidence = quality_and_evidence_for_card(card, session, events)
    return quality


def quality_and_evidence_for_card(
    card: ImprovementCard,
    session: PracticeSession,
    events: Sequence,
) -> tuple[float | None, dict | None]:
    if card.kind == ImprovementCard.KIND_FLUENCY:
        return _fluency_quality(session), None
    if card.kind == ImprovementCard.KIND_PHRASE:
        return None, None

    evidence = card_evidence(card, session, events)
    if evidence is None or int(evidence.get("opportunities", 0)) <= 0:
        return None, evidence

    opportunities = int(evidence["opportunities"])
    misses = int(evidence.get("misses", 0))
    target_quality = 1.0 - (float(misses) / float(opportunities))
    target_quality = max(0.0, min(1.0, target_quality))
    w = float(opportunities) / float(opportunities + SHRINKAGE_K)
    quality = (w * target_quality) + ((1.0 - w) * session_quality(session))
    return round(quality, 3), evidence


def _word_evidence(target_key: str, session: PracticeSession, events: Sequence) -> dict:
    target = clean_text_for_alignment(target_key).strip()
    tokens = clean_text_for_alignment(session.script_text).split()
    opportunities = sum(1 for token in tokens if token == target)
    misses = 0
    for row in events:
        if row.op not in {"sub", "del"}:
            continue
        if clean_text_for_alignment(row.ref_token or "").strip() == target:
            misses += 1
    return {"opportunities": opportunities, "misses": misses}


def _sound_evidence(target_key: str, session: PracticeSession, events: Sequence) -> dict:
    target = str(target_key or "").strip().upper()
    stats, _top_pairs = _phoneme_stats([session], events)
    row = stats.get(target)
    if not row:
        return {"opportunities": 0, "misses": 0}
    return {
        "opportunities": int(row.get("attempts", 0)),
        "misses": int(row.get("errors", 0)),
    }


def _character_evidence(target_key: str, session: PracticeSession, events: Sequence) -> dict:
    target = clean_text_for_alignment(target_key).replace(" ", "")
    target = target[:1]
    text = clean_text_for_alignment(session.script_text).replace(" ", "")
    opportunities = sum(1 for ch in text if ch == target) if target else 0

    misses = 0
    for row in events:
        if not str(row.error_kind or "").startswith("char_"):
            continue
        ref_token = clean_text_for_alignment(row.ref_token or "").replace(" ", "")
        if not ref_token or not target:
            continue
        start = int(row.ref_local_start) if row.ref_local_start is not None else 0
        end = int(row.ref_local_end) if row.ref_local_end is not None else len(ref_token)
        start = max(0, min(start, len(ref_token)))
        end = max(start, min(end, len(ref_token)))
        ref_chunk = ref_token[start:end] or ref_token
        if target in ref_chunk:
            misses += 1
    return {"opportunities": opportunities, "misses": misses}


def _position_evidence(target_key: str, session: PracticeSession, events: Sequence) -> dict:
    target = str(target_key or "").strip().lower()
    stats = _position_stats([session], events)
    row = stats.get(target)
    if not row:
        return {"opportunities": 0, "misses": 0}

    # Keep miss attribution tied to the same helper used by position trends.
    misses = 0
    for event in events:
        if event.op not in ("sub", "del"):
            continue
        bucket = _position_bucket(event.ref_local_start, event.ref_local_end, event.ref_token_len)
        if bucket == target:
            misses += 1
    return {
        "opportunities": int(row.get("attempts", 0)),
        "misses": misses,
    }


def _fluency_quality(session: PracticeSession) -> float:
    components = []
    if session.artic_rate is not None:
        components.append(
            _band_quality(
                float(session.artic_rate),
                ARTIC_RATE_IDEAL_MIN,
                ARTIC_RATE_IDEAL_MAX,
                ARTIC_RATE_DECAY,
            )
        )
    if session.pause_ratio is not None:
        components.append(
            _band_quality(
                float(session.pause_ratio),
                PAUSE_RATIO_IDEAL_MIN,
                PAUSE_RATIO_IDEAL_MAX,
                PAUSE_RATIO_DECAY,
            )
        )
    if session.filled_pauses is not None:
        filled_quality = 1.0 - min(1.0, max(0.0, float(session.filled_pauses)) / FILLED_PAUSES_DECAY)
        components.append(filled_quality)
    if not components:
        return session_quality(session)
    return round(sum(components) / len(components), 3)


def _band_quality(value: float, low: float, high: float, decay: float) -> float:
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return max(0.0, 1.0 - min(1.0, distance / decay))
