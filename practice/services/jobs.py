from __future__ import annotations

import threading
import uuid
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import close_old_connections, transaction
from django.utils import timezone

from practice.models import GeneratedPracticeScript, ImprovementCard, PracticeScript, ScoringJob, default_practice_user_pk
from practice.services.audio_storage import materialized_audio
from practice.services.analytics import refresh_improvement_cards
from practice.services.scoring import transcribe_free_and_store, transcribe_score_and_store
from practice.services.spaced_repetition import update_card_from_session


def create_scoring_job(
    *,
    user=None,
    script: PracticeScript,
    audio_path: str,
    provider: str,
    card: ImprovementCard | None = None,
    submission_id: uuid.UUID | None = None,
) -> ScoringJob:
    if user is None:
        user = get_user_model().objects.get(pk=default_practice_user_pk())
    linked_card = card or _card_for_script(script)
    return ScoringJob.objects.create(
        user=user,
        script=script,
        submission_id=submission_id,
        card=linked_card,
        script_name=script.title,
        script_text=script.body,
        audio_path=audio_path,
        provider=provider,
        mode=ScoringJob.MODE_SCORE,
    )


def create_free_speak_job(
    *,
    user=None,
    audio_path: str,
    provider: str,
    submission_id: uuid.UUID | None = None,
) -> ScoringJob:
    if user is None:
        user = get_user_model().objects.get(pk=default_practice_user_pk())
    return ScoringJob.objects.create(
        user=user,
        script=None,
        submission_id=submission_id,
        card=None,
        script_name="Free Speak",
        script_text="",
        audio_path=audio_path,
        provider=provider,
        mode=ScoringJob.MODE_FREE,
    )


def enqueue_scoring_job(job: ScoringJob) -> None:
    mode = "inline" if settings.SCORING_JOBS_INLINE else settings.SCORING_JOBS_MODE
    if mode == "inline":
        process_scoring_job(job.pk)
        return
    if mode == "queue":
        return
    if mode != "thread":
        raise ValueError(f"Unknown SCORING_JOBS_MODE: {mode}")

    thread = threading.Thread(
        target=_process_job_in_thread,
        args=(job.pk,),
        name=f"speechpractice-score-{job.pk}",
        daemon=True,
    )
    thread.start()


def _process_job_in_thread(job_id: int) -> None:
    close_old_connections()
    try:
        process_scoring_job(job_id)
    finally:
        close_old_connections()


def process_scoring_job(job_id: int) -> ScoringJob:
    with transaction.atomic():
        job = ScoringJob.objects.select_for_update().get(pk=job_id)
        if job.status not in {ScoringJob.STATUS_QUEUED, ScoringJob.STATUS_FAILED}:
            return job
        job.status = ScoringJob.STATUS_RUNNING
        job.started_at = timezone.now()
        job.error_message = ""
        job.save(update_fields=["status", "started_at", "error_message", "updated_at"])

    def on_partial(text: str) -> None:
        ScoringJob.objects.filter(pk=job_id).update(
            partial_transcript=str(text or "")[:20000],
            updated_at=timezone.now(),
        )

    try:
        with materialized_audio(job.audio_path) as local_audio_path:
            if job.mode == ScoringJob.MODE_FREE:
                session = transcribe_free_and_store(
                    user=job.user,
                    audio_path=local_audio_path,
                    stored_audio_ref=job.audio_path,
                    provider_name=job.provider,
                    partial_callback=on_partial,
                )
            else:
                session = transcribe_score_and_store(
                    user=job.user,
                    script_name=job.script_name,
                    script_text=job.script_text,
                    audio_path=local_audio_path,
                    stored_audio_ref=job.audio_path,
                    provider_name=job.provider,
                    partial_callback=on_partial,
                )
                update_card_from_session(job.card, session)
                refresh_improvement_cards(user=job.user, days=settings.CARD_REFRESH_WINDOW_DAYS)
    except Exception as exc:
        job = ScoringJob.objects.get(pk=job_id)
        job.status = ScoringJob.STATUS_FAILED
        job.error_message = str(exc)
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        return job

    job = ScoringJob.objects.get(pk=job_id)
    job.status = ScoringJob.STATUS_SUCCEEDED
    job.legacy_session_id = session.pk
    job.partial_transcript = session.transcript or job.partial_transcript
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "legacy_session_id", "partial_transcript", "finished_at", "updated_at"])
    return job


def process_next_scoring_job() -> ScoringJob | None:
    job = ScoringJob.objects.filter(status=ScoringJob.STATUS_QUEUED).order_by("created_at").first()
    if job is None:
        return None
    return process_scoring_job(job.pk)


def recover_stale_scoring_jobs(*, stale_after_minutes: int = 30) -> int:
    cutoff = timezone.now() - timedelta(minutes=max(1, stale_after_minutes))
    return ScoringJob.objects.filter(
        status=ScoringJob.STATUS_RUNNING,
        started_at__lt=cutoff,
    ).update(
        status=ScoringJob.STATUS_QUEUED,
        started_at=None,
        finished_at=None,
        error_message="Recovered after the scoring worker stopped unexpectedly.",
        updated_at=timezone.now(),
    )


def _card_for_script(script: PracticeScript) -> ImprovementCard | None:
    generated_query = GeneratedPracticeScript.objects.select_related("card").filter(
        script=script,
        card__isnull=False,
    )
    if script.user_id is not None:
        generated_query = generated_query.filter(user=script.user)
    generated = generated_query.order_by("-created_at").first()
    if generated and generated.card:
        return generated.card
    if script.source_ref.startswith("card:"):
        try:
            card_query = ImprovementCard.objects.filter(pk=int(script.source_ref.split(":", 1)[1]))
            if script.user_id is not None:
                card_query = card_query.filter(user=script.user)
            return card_query.get()
        except (ImprovementCard.DoesNotExist, ValueError):
            return None
    return None


def job_status_context(job: ScoringJob) -> dict[str, Any]:
    return {
        "is_pending": job.status in {ScoringJob.STATUS_QUEUED, ScoringJob.STATUS_RUNNING},
        "is_done": job.status == ScoringJob.STATUS_SUCCEEDED,
        "is_failed": job.status == ScoringJob.STATUS_FAILED,
    }
