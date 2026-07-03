from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Avg, Count
from django.utils import timezone

from error_analytics import (
    character_trend_summary,
    generate_feedback_summary,
    phoneme_trend_summary,
    phrase_trend_summary,
    position_trend_summary,
    window_bounds,
    word_trend_summary,
)

from practice.models import (
    GeneratedPracticeScript,
    ImprovementCard,
    PracticeReview,
    PracticeScript,
    PracticeSession,
    ScoringJob,
    SessionError,
    default_practice_user_pk,
)


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


def _resolve_user(user=None):
    if user is not None and getattr(user, "is_authenticated", True):
        return user
    return get_user_model().objects.get(pk=default_practice_user_pk())


def dashboard_stats(user=None) -> DashboardStats:
    user = _resolve_user(user)
    base = PracticeSession.objects.filter(user=user)
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


def recent_sessions(user=None, limit: int = 8):
    user = _resolve_user(user)
    return PracticeSession.objects.filter(user=user).exclude(score__isnull=True).order_by("-timestamp", "-id")[:limit]


def today_queue(user=None, limit: int = 5) -> list[TodayQueueItem]:
    user = _resolve_user(user)
    now = timezone.now()
    cards = list(
        ImprovementCard.objects.filter(user=user)
        .exclude(status=ImprovementCard.STATUS_PAUSED)
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


def active_scoring_jobs(user=None, limit: int = 4):
    user = _resolve_user(user)
    return ScoringJob.objects.filter(
        user=user,
        status__in=[ScoringJob.STATUS_QUEUED, ScoringJob.STATUS_RUNNING]
    ).order_by("-created_at")[:limit]


def recent_scoring_jobs(user=None, limit: int = 4):
    user = _resolve_user(user)
    return ScoringJob.objects.filter(
        user=user,
        status__in=[ScoringJob.STATUS_SUCCEEDED, ScoringJob.STATUS_FAILED]
    ).order_by("-finished_at", "-created_at")[:limit]


def _metric_tag(metric: str, value: float) -> tuple[str, str]:
    if metric == "wer":
        if value <= 0.05:
            return "Excellent", "good"
        if value <= 0.10:
            return "Good", "ok"
        return "Needs work", "watch"
    if metric == "cer":
        if value <= 0.02:
            return "Excellent", "good"
        if value <= 0.04:
            return "Good", "ok"
        return "Needs work", "watch"
    if metric == "clarity":
        if value >= 0.95:
            return "Excellent", "good"
        if value >= 0.90:
            return "Good", "ok"
        return "Watch", "watch"
    if metric == "avg_conf":
        if value >= 0.85:
            return "Excellent", "good"
        if value >= 0.75:
            return "Good", "ok"
        return "Watch", "watch"
    if metric == "artic_rate":
        if 120 <= value <= 160:
            return "On target", "good"
        if 100 <= value < 120 or 160 < value <= 170:
            return "Near target", "ok"
        return "Watch", "watch"
    if metric == "pause_ratio":
        if 0.10 <= value <= 0.25:
            return "On target", "good"
        if 0.25 < value <= 0.35:
            return "OK", "ok"
        return "Watch", "watch"
    return "", "ok"


_HOME_METRICS = [
    ("wer", "WER", "pct1"),
    ("cer", "CER", "pct1"),
    ("clarity", "Clarity", "pct0"),
    ("avg_conf", "Avg confidence", "conf"),
    ("artic_rate", "Articulation", "wpm"),
    ("pause_ratio", "Pause ratio", "pct0"),
]


def home_snapshot(user=None, queue: list[TodayQueueItem] | None = None) -> dict[str, Any]:
    """Streak, weekly metric bands, and clarity trend for the dashboard hero."""
    user = _resolve_user(user)
    queue = queue or []
    now = datetime.now()
    rows = PracticeSession.objects.filter(user=user).exclude(score__isnull=True).values_list(
        "timestamp", "wer", "cer", "clarity", "avg_conf", "artic_rate", "pause_ratio"
    )
    dated: list[tuple[datetime, tuple]] = []
    for ts, *vals in rows:
        dt = _parse_session_timestamp(ts)
        if dt is not None:
            dated.append((dt, tuple(vals)))
    dated.sort(key=lambda item: item[0])

    practice_days = {dt.date() for dt, _ in dated}
    today = now.date()
    streak_days = 0
    cursor = today if today in practice_days else today - timedelta(days=1)
    while cursor in practice_days:
        streak_days += 1
        cursor -= timedelta(days=1)
    best_streak = 0
    run = 0
    previous = None
    for day in sorted(practice_days):
        run = run + 1 if previous is not None and day - previous == timedelta(days=1) else 1
        best_streak = max(best_streak, run)
        previous = day

    reps_today = sum(1 for dt, _ in dated if dt.date() == today)
    due_count = sum(1 for item in queue if item.is_due)
    daily_goal = max(reps_today + due_count, 1)
    ring_fraction = min(1.0, reps_today / daily_goal)
    circumference = 452.4  # 2 * pi * r for the r=72 ring
    ring_offset = round(circumference * (1 - ring_fraction), 1)

    def _averages(entries: list[tuple]) -> dict[str, float]:
        out: dict[str, float] = {}
        for index, (key, _label, _fmt) in enumerate(_HOME_METRICS):
            values = [row[index] for row in entries if row[index] is not None]
            if values:
                out[key] = sum(values) / len(values)
        return out

    week_rows = [vals for dt, vals in dated if dt >= now - timedelta(days=7)]
    prev_rows = [vals for dt, vals in dated if now - timedelta(days=14) <= dt < now - timedelta(days=7)]
    week_avg = _averages(week_rows or [vals for _, vals in dated[-6:]])
    prev_avg = _averages(prev_rows)

    metrics = []
    for key, label, fmt in _HOME_METRICS:
        value = week_avg.get(key)
        if value is None:
            continue
        if fmt == "pct1":
            display, unit = f"{value * 100:.1f}", "%"
        elif fmt == "pct0":
            display, unit = f"{value * 100:.0f}", "%"
        elif fmt == "wpm":
            display, unit = f"{value:.0f}", "wpm"
        else:
            display, unit = f"{value:.2f}", ""
        tag_label, tag_class = _metric_tag(key, value)
        metrics.append(
            {"label": label, "display": display, "unit": unit, "tag": tag_label, "tag_class": tag_class}
        )

    clarity_now = week_avg.get("clarity")
    clarity_prev = prev_avg.get("clarity")
    clarity_delta = None
    if clarity_now is not None and clarity_prev is not None:
        clarity_delta = round((clarity_now - clarity_prev) * 100)

    spark_values = [vals[2] for _, vals in dated if vals[2] is not None][-8:]
    spark_points = None
    spark_last = None
    if len(spark_values) >= 2:
        low, high = min(spark_values), max(spark_values)
        spread = (high - low) or 1.0
        coords = []
        for index, value in enumerate(spark_values):
            x = round(index * 180 / (len(spark_values) - 1), 1)
            y = round(38 - ((value - low) / spread) * 32, 1)
            coords.append(f"{x},{y}")
        spark_points = " ".join(coords)
        spark_last = coords[-1].split(",")

    return {
        "streak_days": streak_days,
        "best_streak": best_streak,
        "reps_today": reps_today,
        "daily_goal": daily_goal,
        "ring_offset": ring_offset,
        "metrics": metrics,
        "clarity_now": round(clarity_now * 100) if clarity_now is not None else None,
        "clarity_delta": clarity_delta,
        "clarity_delta_abs": abs(clarity_delta) if clarity_delta is not None else None,
        "spark_points": spark_points,
        "spark_last_x": spark_last[0] if spark_last else None,
        "spark_last_y": spark_last[1] if spark_last else None,
        "queued_count": len(queue),
        "due_count": due_count,
        "est_minutes": 3 * len(queue),
    }


def _latest_generated_scripts(cards: list[ImprovementCard]) -> dict[int, PracticeScript]:
    card_ids = [card.pk for card in cards]
    if not card_ids:
        return {}
    rows = (
        GeneratedPracticeScript.objects.select_related("script")
        .filter(
            card_id__in=card_ids,
            script__active=True,
            script__practice_kind=PracticeScript.KIND_DRILL,
        )
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


def trend_summary(user=None, days: int = 30) -> dict[str, Any]:
    user = _resolve_user(user)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    return trend_summary_for_range(user=user, start_dt=start_dt, end_dt=end_dt)


def _fetch_trend_windows(
    user,
    recent_start: datetime,
    recent_end: datetime,
    prev_start: datetime | None,
    prev_end: datetime | None,
    script_name: str | None = None,
) -> tuple[list, list, list | None, list | None]:
    """Load scored sessions and their error events for both trend windows.

    One session query and one event query cover all five trend summaries.
    Returns (recent_sessions, recent_events, prev_sessions, prev_events);
    the prev pair is None when there is no previous window.
    """
    fetch_start = prev_start if prev_start is not None else recent_start
    sessions_qs = (
        PracticeSession.objects.filter(user=user)
        .exclude(script_text="")
        .exclude(transcript__isnull=True)
        .exclude(transcript="")
        .only("id", "timestamp", "script_name", "script_text")
    )
    if script_name and script_name != "All scripts":
        sessions_qs = sessions_qs.filter(script_name=script_name)
    # Coarse SQL prefilter on the ISO date prefix; timestamps are strings, so
    # exact bounds are applied in Python after parsing.
    sessions_qs = sessions_qs.filter(timestamp__gte=fetch_start.date().isoformat())

    recent_sessions: list = []
    prev_sessions: list = []
    for sess in sessions_qs:
        parsed = _parse_session_timestamp(sess.timestamp)
        if parsed is None:
            continue
        if recent_start <= parsed <= recent_end:
            recent_sessions.append(sess)
        elif prev_start is not None and prev_start <= parsed <= prev_end:
            prev_sessions.append(sess)

    session_ids = [s.id for s in recent_sessions] + [s.id for s in prev_sessions]
    events: list = []
    if session_ids:
        events = list(
            SessionError.objects.filter(session_id__in=session_ids).only(
                "id",
                "session_id",
                "op",
                "error_kind",
                "ref_token",
                "hyp_token",
                "ref_start",
                "ref_end",
                "ref_local_start",
                "ref_local_end",
                "hyp_local_start",
                "hyp_local_end",
                "ref_token_len",
            )
        )
    recent_ids = {s.id for s in recent_sessions}
    recent_events = [e for e in events if e.session_id in recent_ids]
    prev_events = [e for e in events if e.session_id not in recent_ids]

    if prev_start is None:
        return recent_sessions, recent_events, None, None
    return recent_sessions, recent_events, prev_sessions, prev_events


def trend_summary_for_range(
    *,
    user=None,
    start_dt: datetime,
    end_dt: datetime,
    script_name: str | None = None,
) -> dict[str, Any]:
    user = _resolve_user(user)
    recent_start, recent_end, prev_start, prev_end = window_bounds(start_dt, end_dt)
    recent_sessions, recent_events, prev_sessions, prev_events = _fetch_trend_windows(
        user,
        recent_start,
        recent_end,
        prev_start,
        prev_end,
        script_name=script_name,
    )
    words = word_trend_summary(
        recent_sessions, recent_events, prev_sessions, prev_events, top_n=6, min_attempts=2
    )
    chars = character_trend_summary(
        recent_sessions, recent_events, prev_sessions, prev_events, top_n=5
    )
    positions = position_trend_summary(
        recent_sessions, recent_events, prev_sessions, prev_events, top_n=4
    )
    sounds = phoneme_trend_summary(
        recent_sessions, recent_events, prev_sessions, prev_events, top_n=5, min_attempts=3
    )
    phrases = phrase_trend_summary(
        recent_sessions, recent_events, prev_sessions, prev_events, top_n=5, min_attempts=1
    )
    return {
        "words": words,
        "characters": chars,
        "positions": positions,
        "sounds": sounds,
        "phrases": phrases,
        "feedback": generate_feedback_summary(words, chars, positions, sounds, phrases),
    }


def script_name_options(user=None) -> list[str]:
    user = _resolve_user(user)
    rows = (
        PracticeSession.objects.filter(user=user)
        .exclude(script_name="")
        .values_list("script_name", flat=True)
        .distinct()
    )
    return sorted({str(name) for name in rows if name})


def progress_series(
    *,
    user=None,
    start_dt: datetime,
    end_dt: datetime,
    script_name: str | None = None,
) -> list[dict[str, Any]]:
    user = _resolve_user(user)
    rows = PracticeSession.objects.filter(user=user).exclude(score__isnull=True)
    if script_name:
        rows = rows.filter(script_name=script_name)
    points = []
    for session in rows:
        parsed = _parse_session_timestamp(session.timestamp)
        if parsed is None or parsed < start_dt or parsed > end_dt:
            continue
        points.append(
            {
                "date": parsed.isoformat(),
                "label": parsed.strftime("%m/%d"),
                "script": session.script_name,
                "score": session.score,
                "wer": session.wer,
                "clarity": session.clarity,
                "cer": session.cer,
                "artic_rate": session.artic_rate,
                "pause_ratio": session.pause_ratio,
            }
        )
    points.sort(key=lambda item: item["date"])
    return points


def _parse_session_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


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
                "title": str(word),
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
                "title": str(symbol),
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
                "title": str(phrase),
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
                "title": str(bucket),
                "prompt": f"Practice crisp articulation for sounds at the {bucket} of words.",
                "stats": row,
                "mastery": _mastery_from_error_rate(rate),
            }
        )

    return candidates


def refresh_improvement_cards(
    user=None,
    days: int | None = None,
    *,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> int:
    user = _resolve_user(user)
    if start_dt is not None and end_dt is not None:
        summary = trend_summary_for_range(user=user, start_dt=start_dt, end_dt=end_dt)
        source_window = {
            "source_window_start": start_dt.date().isoformat(),
            "source_window_end": end_dt.date().isoformat(),
            "source_window_label": f"{start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}",
        }
    else:
        window_days = int(days or settings.CARD_REFRESH_WINDOW_DAYS)
        summary = trend_summary(user=user, days=window_days)
        source_window = {
            "source_window_days": window_days,
            "source_window_label": _source_window_label(window_days),
        }
    refreshed_at = timezone.now()
    count = 0
    for candidate in build_card_candidates(summary):
        mastery = float(candidate["mastery"])
        status = (
            ImprovementCard.STATUS_REVIEW
            if mastery >= 0.65
            else ImprovementCard.STATUS_LEARNING
        )
        stats = dict(candidate["stats"])
        stats.update(source_window)
        stats["source_window_refreshed_at"] = refreshed_at.isoformat()
        _card, created = ImprovementCard.objects.update_or_create(
            user=user,
            kind=candidate["kind"],
            target_key=candidate["target_key"],
            defaults={
                "title": candidate["title"],
                "prompt": candidate["prompt"],
                "stats": stats,
                "mastery": mastery,
                "status": status,
                "due_at": _due_date_for_mastery(mastery),
            },
        )
        count += 1 if created else 0
    return count


def _source_window_label(days: int) -> str:
    if days >= 3650:
        return "all history"
    if days >= 365:
        years = round(days / 365)
        return f"last {years} year" if years == 1 else f"last {years} years"
    return f"last {days} days"


def due_cards(user=None, limit: int = 6):
    user = _resolve_user(user)
    return ImprovementCard.objects.filter(user=user).exclude(status=ImprovementCard.STATUS_PAUSED).filter(
        due_at__lte=timezone.now()
    )[:limit]


def score_distribution(user=None) -> dict[str, int]:
    user = _resolve_user(user)
    scored = PracticeSession.objects.filter(user=user).exclude(score__isnull=True)
    return {
        "needs_work": scored.filter(score__lt=3).aggregate(count=Count("id"))["count"] or 0,
        "solid": scored.filter(score__gte=3, score__lt=4.25).aggregate(count=Count("id"))["count"] or 0,
        "strong": scored.filter(score__gte=4.25).aggregate(count=Count("id"))["count"] or 0,
    }
