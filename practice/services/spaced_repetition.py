from __future__ import annotations

from datetime import timedelta

from django.utils import timezone

from practice.models import ImprovementCard, PracticeReview, PracticeSession


def quality_from_session(session: PracticeSession) -> float:
    score = float(session.score or 0.0)
    wer = float(session.wer or 1.0)
    score_quality = max(0.0, min(1.0, (score - 1.0) / 4.0))
    wer_quality = max(0.0, min(1.0, 1.0 - wer))
    return round((score_quality * 0.65) + (wer_quality * 0.35), 3)


def update_card_from_session(
    card: ImprovementCard | None,
    session: PracticeSession,
) -> PracticeReview | None:
    if card is None:
        return None

    quality = quality_from_session(session)
    previous_mastery = float(card.mastery or 0.0)
    new_mastery = _next_mastery(previous_mastery, quality)
    card.mastery = new_mastery
    card.status = _status_for_mastery(new_mastery)
    card.due_at = timezone.now() + _interval_for_mastery(new_mastery, quality)
    card.last_reviewed_at = timezone.now()
    card.stats = {
        **(card.stats or {}),
        "last_review": {
            "legacy_session_id": session.pk,
            "score": float(session.score or 0.0),
            "wer": float(session.wer or 0.0),
            "quality": quality,
            "previous_mastery": previous_mastery,
            "mastery": new_mastery,
        },
    }
    card.save(
        update_fields=[
            "mastery",
            "status",
            "due_at",
            "last_reviewed_at",
            "stats",
            "updated_at",
        ]
    )
    return PracticeReview.objects.create(
        user=session.user,
        card=card,
        legacy_session_id=session.pk,
        score=session.score,
        error_rate=session.wer,
        notes=(
            f"Auto review from scored session. Quality {quality:.2f}; "
            f"mastery {previous_mastery:.2f} -> {new_mastery:.2f}."
        ),
    )


def _next_mastery(previous: float, quality: float) -> float:
    if quality >= 0.82:
        adjusted = previous + ((1.0 - previous) * 0.42)
    elif quality >= 0.62:
        adjusted = previous + ((quality - previous) * 0.28)
    else:
        adjusted = previous - ((0.62 - quality) * 0.32)
    return round(max(0.0, min(1.0, adjusted)), 3)


def _status_for_mastery(mastery: float) -> str:
    if mastery >= 0.92:
        return ImprovementCard.STATUS_MASTERED
    if mastery >= 0.62:
        return ImprovementCard.STATUS_REVIEW
    return ImprovementCard.STATUS_LEARNING


def _interval_for_mastery(mastery: float, quality: float) -> timedelta:
    if quality < 0.5:
        return timedelta(days=1)
    if mastery >= 0.92:
        return timedelta(days=21)
    if mastery >= 0.8:
        return timedelta(days=10)
    if mastery >= 0.62:
        return timedelta(days=4)
    return timedelta(days=2)
