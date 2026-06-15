from __future__ import annotations

import threading
from typing import Any

from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils import timezone

from practice.models import GeneratedPracticeScript, ImprovementCard, PracticeScript, ScoringJob
from practice.services.analytics import refresh_improvement_cards
from practice.services.scoring import transcribe_free_and_store, transcribe_score_and_store
from practice.services.spaced_repetition import update_card_from_session


def create_scoring_job(
    *,
    script: PracticeScript,
    audio_path: str,
    provider: str,
    card: ImprovementCard | None = None,
) -> ScoringJob:
    linked_card = card or _card_for_script(script)
    return ScoringJob.objects.create(
        script=script,
        card=linked_card,
        script_name=script.title,
        script_text=script.body,
        audio_path=audio_path,
        provider=provider,
        mode=ScoringJob.MODE_SCORE,
    )


def create_free_speak_job(
    *,
    audio_path: str,
    provider: str,
) -> ScoringJob:
    return ScoringJob.objects.create(
        script=None,
        card=None,
        script_name="Free Speak",
        script_text="",
        audio_path=audio_path,
        provider=provider,
        mode=ScoringJob.MODE_FREE,
    )


def enqueue_scoring_job(job: ScoringJob) -> None:
    if settings.SCORING_JOBS_INLINE:
        process_scoring_job(job.pk)
        return

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
        if job.mode == ScoringJob.MODE_FREE:
            session = transcribe_free_and_store(
                audio_path=job.audio_path,
                provider_name=job.provider,
                partial_callback=on_partial,
            )
        else:
            session = transcribe_score_and_store(
                script_name=job.script_name,
                script_text=job.script_text,
                audio_path=job.audio_path,
                provider_name=job.provider,
                partial_callback=on_partial,
            )
            update_card_from_session(job.card, session)
            refresh_improvement_cards(days=settings.CARD_REFRESH_WINDOW_DAYS)
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


def _card_for_script(script: PracticeScript) -> ImprovementCard | None:
    generated = (
        GeneratedPracticeScript.objects.select_related("card")
        .filter(script=script, card__isnull=False)
        .order_by("-created_at")
        .first()
    )
    if generated and generated.card:
        return generated.card
    if script.source_ref.startswith("card:"):
        try:
            return ImprovementCard.objects.get(pk=int(script.source_ref.split(":", 1)[1]))
        except (ImprovementCard.DoesNotExist, ValueError):
            return None
    return None


def job_status_context(job: ScoringJob) -> dict[str, Any]:
    return {
        "is_pending": job.status in {ScoringJob.STATUS_QUEUED, ScoringJob.STATUS_RUNNING},
        "is_done": job.status == ScoringJob.STATUS_SUCCEEDED,
        "is_failed": job.status == ScoringJob.STATUS_FAILED,
    }
