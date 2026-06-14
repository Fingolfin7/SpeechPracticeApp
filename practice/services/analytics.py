from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from django.conf import settings
from django.db.models import Avg, Count
from django.utils import timezone

import db as legacy_db
from error_analytics import (
    generate_feedback_summary,
    get_character_trend_summary,
    get_phoneme_trend_summary,
    get_phrase_trend_summary,
    get_position_trend_summary,
    get_word_trend_summary,
)

from practice.models import GeneratedPracticeScript, ImprovementCard, PracticeReview, PracticeScript, PracticeSession, ScoringJob


@dataclass(frozen=True)
class DashboardStats:
    session_count: int
    scored_count: int
    average_score: float | None
    average_wer: float | None
    average_clarity: float | None


@dataclass(frozen=True)
class TodayQueueItem:
    card: ImprovementCard
    script: PracticeScript | None
    review_count: int
    is_due: bool


def dashboard_stats() -> DashboardStats:
    base = PracticeSession.objects.all()
    scored = base.exclude(score__isnull=True)
    aggregate = scored.aggregate(
        average_score=Avg("score"),
        average_wer=Avg("wer"),
        average_clarity=Avg("clarity"),
    )
    return DashboardStats(
        session_count=base.count(),
        scored_count=scored.count(),
        average_score=aggregate["average_score"],
        average_wer=aggregate["average_wer"],
        average_clarity=aggregate["average_clarity"],
    )


def recent_sessions(limit: int = 8):
    return PracticeSession.objects.exclude(score__isnull=True).order_by("-timestamp", "-id")[:limit]


def today_queue(limit: int = 5) -> list[TodayQueueItem]:
    now = timezone.now()
    cards = list(
        ImprovementCard.objects.exclude(status=ImprovementCard.STATUS_PAUSED)
        .order_by("due_at", "-updated_at")[: max(limit * 3, limit)]
    )
    cards.sort(key=lambda card: (card.due_at > now, card.due_at, -card.mastery))
    selected = cards[:limit]
    scripts_by_card = _latest_generated_scripts(selected)
    review_counts = _review_counts(selected)
    return [
        TodayQueueItem(
            card=card,
            script=scripts_by_card.get(card.pk),
            review_count=review_counts.get(card.pk, 0),
            is_due=card.due_at <= now,
        )
        for card in selected
    ]


def active_scoring_jobs(limit: int = 4):
    return ScoringJob.objects.filter(
        status__in=[ScoringJob.STATUS_QUEUED, ScoringJob.STATUS_RUNNING]
    ).order_by("-created_at")[:limit]


def recent_scoring_jobs(limit: int = 4):
    return ScoringJob.objects.filter(
        status__in=[ScoringJob.STATUS_SUCCEEDED, ScoringJob.STATUS_FAILED]
    ).order_by("-finished_at", "-created_at")[:limit]


def _latest_generated_scripts(cards: list[ImprovementCard]) -> dict[int, PracticeScript]:
    card_ids = [card.pk for card in cards]
    if not card_ids:
        return {}
    rows = (
        GeneratedPracticeScript.objects.select_related("script")
        .filter(card_id__in=card_ids, script__active=True)
        .order_by("card_id", "-created_at")
    )
    scripts: dict[int, PracticeScript] = {}
    for row in rows:
        if row.card_id not in scripts and row.script is not None:
            scripts[row.card_id] = row.script
    return scripts


def _review_counts(cards: list[ImprovementCard]) -> dict[int, int]:
    card_ids = [card.pk for card in cards]
    if not card_ids:
        return {}
    return {
        row["card_id"]: row["count"]
        for row in PracticeReview.objects.filter(card_id__in=card_ids)
        .values("card_id")
        .annotate(count=Count("id"))
    }


def trend_summary(days: int = 30) -> dict[str, Any]:
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    session = legacy_db.get_session(str(settings.LEGACY_DB_PATH))
    try:
        words = get_word_trend_summary(session, start_dt=start_dt, end_dt=end_dt, top_n=6, min_attempts=2)
        chars = get_character_trend_summary(session, start_dt=start_dt, end_dt=end_dt, top_n=5)
        positions = get_position_trend_summary(session, start_dt=start_dt, end_dt=end_dt, top_n=4)
        sounds = get_phoneme_trend_summary(session, start_dt=start_dt, end_dt=end_dt, top_n=5, min_attempts=3)
        phrases = get_phrase_trend_summary(session, start_dt=start_dt, end_dt=end_dt, top_n=5, min_attempts=1)
        return {
            "words": words,
            "characters": chars,
            "positions": positions,
            "sounds": sounds,
            "phrases": phrases,
            "feedback": generate_feedback_summary(words, chars, positions, sounds, phrases),
        }
    finally:
        session.close()


def _mastery_from_error_rate(error_rate: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - float(error_rate))), 3)


def _due_date_for_mastery(mastery: float):
    if mastery >= 0.9:
        days = 14
    elif mastery >= 0.75:
        days = 7
    elif mastery >= 0.55:
        days = 3
    else:
        days = 1
    return timezone.now() + timedelta(days=days)


def build_card_candidates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for row in summary.get("words", {}).get("top_trouble_words", [])[:8]:
        word = row.get("word")
        if not word:
            continue
        rate = float(row.get("error_rate", 0.0))
        candidates.append(
            {
                "kind": ImprovementCard.KIND_WORD,
                "target_key": str(word),
                "title": f"Word focus: {word}",
                "prompt": f"Practice short phrases that place '{word}' at the beginning, middle, and end of a sentence.",
                "stats": row,
                "mastery": _mastery_from_error_rate(rate),
            }
        )

    for row in summary.get("sounds", {}).get("top_trouble_symbols", [])[:6]:
        symbol = row.get("symbol")
        if not symbol:
            continue
        rate = float(row.get("error_rate", 0.0))
        candidates.append(
            {
                "kind": ImprovementCard.KIND_SOUND,
                "target_key": str(symbol),
                "title": f"Sound pattern: {symbol}",
                "prompt": f"Generate minimal-pair and sentence drills that emphasize the {symbol} sound.",
                "stats": row,
                "mastery": _mastery_from_error_rate(rate),
            }
        )

    for row in summary.get("phrases", {}).get("top_trouble_phrases", [])[:5]:
        phrase = row.get("phrase")
        if not phrase:
            continue
        rate = float(row.get("error_rate", 0.0))
        candidates.append(
            {
                "kind": ImprovementCard.KIND_PHRASE,
                "target_key": str(phrase),
                "title": f"Phrase focus: {phrase}",
                "prompt": (
                    "Practice this phrase in isolation, then inside short surrounding "
                    "sentences while keeping pace and articulation steady."
                ),
                "stats": row,
                "mastery": _mastery_from_error_rate(rate),
            }
        )

    for row in summary.get("positions", {}).get("top_position_buckets", [])[:4]:
        bucket = row.get("bucket")
        if not bucket:
            continue
        rate = float(row.get("error_rate", 0.0))
        candidates.append(
            {
                "kind": ImprovementCard.KIND_POSITION,
                "target_key": str(bucket),
                "title": f"Word position: {bucket}",
                "prompt": f"Practice crisp articulation for sounds at the {bucket} of words.",
                "stats": row,
                "mastery": _mastery_from_error_rate(rate),
            }
        )

    return candidates


def refresh_improvement_cards(days: int = 30) -> int:
    summary = trend_summary(days=days)
    count = 0
    for candidate in build_card_candidates(summary):
        mastery = float(candidate["mastery"])
        status = (
            ImprovementCard.STATUS_REVIEW
            if mastery >= 0.65
            else ImprovementCard.STATUS_LEARNING
        )
        _card, created = ImprovementCard.objects.update_or_create(
            kind=candidate["kind"],
            target_key=candidate["target_key"],
            defaults={
                "title": candidate["title"],
                "prompt": candidate["prompt"],
                "stats": candidate["stats"],
                "mastery": mastery,
                "status": status,
                "due_at": _due_date_for_mastery(mastery),
            },
        )
        count += 1 if created else 0
    return count


def due_cards(limit: int = 6):
    return ImprovementCard.objects.exclude(status=ImprovementCard.STATUS_PAUSED).filter(
        due_at__lte=timezone.now()
    )[:limit]


def score_distribution() -> dict[str, int]:
    scored = PracticeSession.objects.exclude(score__isnull=True)
    return {
        "needs_work": scored.filter(score__lt=3).aggregate(count=Count("id"))["count"] or 0,
        "solid": scored.filter(score__gte=3, score__lt=4.25).aggregate(count=Count("id"))["count"] or 0,
        "strong": scored.filter(score__gte=4.25).aggregate(count=Count("id"))["count"] or 0,
    }
