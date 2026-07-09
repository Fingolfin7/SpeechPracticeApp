from __future__ import annotations

import re
import json
import mimetypes
import random
import uuid
from datetime import datetime, timedelta

from django.contrib import messages
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_not_required
from django.conf import settings as django_settings
from django.db import IntegrityError, connection, models, transaction
from django.db.utils import OperationalError, ProgrammingError
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from autumn_client import AutumnClient, AutumnError, normalize_base_url
from error_analytics import (
    _position_bucket,
    _word_to_phoneme_symbols,
    clean_text_for_alignment,
)

from .forms import (
    AccountSettingsForm,
    BulkScriptImportForm,
    PracticeRunForm,
    PracticeScriptForm,
    SelfReviewNotesForm,
    SignUpForm,
    TranscriptEditForm,
)
from .models import (
    GeneratedPracticeScript,
    ImprovementCard,
    LadderStepProgress,
    PracticeLadder,
    PracticeLadderStep,
    PracticeReview,
    PracticeScript,
    PracticeSession,
    PracticeSettings,
    ScoringJob,
    SessionError,
)
from .services.analytics import (
    active_scoring_jobs,
    home_snapshot,
    refresh_improvement_cards,
    progress_series,
    script_name_options,
    today_queue,
    trend_summary_for_range,
    trend_summary,
)
from .services.jobs import create_free_speak_job, create_scoring_job, enqueue_scoring_job, job_status_context
from .services.scoring import _replace_session_errors, score_transcript
from .services.audio_storage import (
    audio_exists as stored_audio_exists,
    audio_size,
    delete_audio,
    open_audio,
    save_uploaded_audio,
)
from .services.codex_auth import (
    CodexAuthError,
    CodexDevicePending,
    deserialize_token_bundle,
    poll_device_code_login,
    serialize_token_bundle,
    start_device_code_login,
    token_bundle_summary,
)
from .services.session_display import audio_exists, highlighted_session_text, session_segments
from .services.script_import import import_script_items, parse_script_upload
from .services.script_generation import (
    generate_cards_from_self_review,
    generate_ladder_draft,
    generate_script_draft,
    script_generation_provider_choices,
)

from .services.transcription import TranscriptResult
from .services.transcription import clear_local_whisper_cache, provider_label

LADDER_MIN_CLARITY_BY_LEVEL = {
    1: 0.85,
    2: 0.88,
    3: 0.90,
    4: 0.92,
    5: 0.95,
}


@login_not_required
def health(request):
    try:
        connection.ensure_connection()
    except Exception:
        return JsonResponse({"ok": False}, status=503)
    return JsonResponse({"ok": True})


@login_not_required
def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            _adopt_placeholder_owner_data(user)
            PracticeSettings.load(user)
            login(request, user)
            messages.success(request, "Welcome to SpeechPractice.")
            return redirect("practice:dashboard")
    else:
        form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})


def _adopt_placeholder_owner_data(user) -> None:
    User = get_user_model()
    placeholder = User.objects.filter(username="owner", is_superuser=True).exclude(pk=user.pk).first()
    if placeholder is None or placeholder.has_usable_password() or User.objects.count() != 2:
        return

    for model in (
        PracticeSession,
        SessionError,
        PracticeScript,
        ImprovementCard,
        PracticeReview,
        ScoringJob,
        GeneratedPracticeScript,
        PracticeLadder,
    ):
        model.objects.filter(user=placeholder).update(user=user)

    existing_settings = PracticeSettings.objects.filter(user=user).first()
    if existing_settings is not None:
        existing_settings.delete()
    PracticeSettings.objects.filter(user=placeholder).update(user=user)
    placeholder.delete()


def dashboard(request):
    summary = trend_summary(request.user)
    queue = today_queue(request.user)
    active_jobs = active_scoring_jobs(request.user)
    now = datetime.now()
    ladders = _practice_ladders(request.user)
    own_ladders = [ladder for ladder in ladders if ladder.user_id == request.user.pk]
    context = {
        "summary": summary,
        "today_queue": queue,
        "active_jobs": active_jobs,
        "home": home_snapshot(request.user, queue),
        "today_label": f"{now:%A}, {now:%B} {now.day}",
        "home_ladders": (own_ladders or ladders)[:1],
    }
    return render(request, "practice/dashboard.html", context)


def progress(request):
    end_dt = _parse_date_param(request.GET.get("end"), datetime.now())
    start_dt = _parse_date_param(request.GET.get("start"), end_dt - timedelta(days=30))
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    script_name = (request.GET.get("script") or "").strip() or None
    points = progress_series(
        user=request.user,
        start_dt=start_dt,
        end_dt=end_dt,
        script_name=script_name,
    )
    summary = trend_summary_for_range(
        user=request.user,
        start_dt=start_dt,
        end_dt=end_dt,
        script_name=script_name,
    )
    return render(
        request,
        "practice/progress.html",
        {
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "selected_script": script_name or "",
            "script_options": script_name_options(request.user),
            "points": points,
            "points_json": json.dumps(points),
            "summary": summary,
        },
    )


def practice_run(request):
    quick_queue = today_queue(request.user, limit=4)
    practice_ladders = _practice_ladders(request.user)
    requested_mode = request.POST.get("mode") if request.method == "POST" else request.GET.get("mode")
    if not requested_mode and request.method == "GET" and request.GET.get("card"):
        requested_mode = PracticeRunForm.MODE_QUICK
    mode = requested_mode or PracticeRunForm.MODE_SCRIPT
    script_kind = _script_kind_for_mode(mode)
    selected_ladder = None
    ladder_steps: list[PracticeLadderStep] = []
    selected_ladder_step = None
    if request.method == "POST":
        submission_id = _submission_id(request)
        if submission_id is not None:
            existing_job = ScoringJob.objects.filter(
                user=request.user,
                submission_id=submission_id,
            ).first()
            if existing_job is not None:
                if _wants_json(request):
                    return JsonResponse(
                        {"ok": True, "message": "Scoring request already received.", **_scoring_job_payload(existing_job)},
                        status=200,
                    )
                return redirect("practice:scoring_job", pk=existing_job.pk)
        form = PracticeRunForm(
            request.POST,
            request.FILES,
            user=request.user,
            script_kind=script_kind,
        )
        if form.is_valid():
            mode = form.cleaned_data.get("mode") or PracticeRunForm.MODE_SCRIPT
            script = form.cleaned_data.get("script")
            card = form.cleaned_data.get("card")
            audio = form.cleaned_data["audio"]
            provider = form.cleaned_data.get("provider") or None
            provider_name = provider or django_settings.TRANSCRIPTION_PROVIDER
            locked_ladder_step = (
                _locked_ladder_step_for_script(request.user, script)
                if mode == PracticeRunForm.MODE_QUICK and script is not None
                else None
            )
            if locked_ladder_step is not None:
                error = "This ladder level is locked. Pass the previous level first."
                form.add_error("script", error)
                if _wants_json(request):
                    return JsonResponse(
                        {
                            "ok": False,
                            "error": error,
                            "errors": form.errors.get_json_data(escape_html=True),
                        },
                        status=400,
                    )
                messages.error(request, error)
            elif audio is None:
                form.add_error("audio", "Record or upload audio before scoring.")
            else:
                audio_path = save_uploaded_audio(
                    audio,
                    script.title if script else "free-speak",
                    user=request.user,
                )
                try:
                    with transaction.atomic():
                        if mode == PracticeRunForm.MODE_FREE:
                            job = create_free_speak_job(
                                user=request.user,
                                audio_path=str(audio_path),
                                provider=provider_name,
                                submission_id=submission_id,
                            )
                        else:
                            job = create_scoring_job(
                                user=request.user,
                                script=script,
                                audio_path=str(audio_path),
                                provider=provider_name,
                                card=card,
                                submission_id=submission_id,
                            )
                except IntegrityError:
                    delete_audio(str(audio_path))
                    job = (
                        ScoringJob.objects.filter(user=request.user, submission_id=submission_id).first()
                        if submission_id is not None
                        else None
                    )
                    if job is None:
                        raise
                enqueue_scoring_job(job)
                focus_label = f" for {card.display_title}" if card else ""
                mode_label = "Free Speak transcription" if mode == PracticeRunForm.MODE_FREE else "Scoring"
                message = f"{mode_label} queued{focus_label} with {provider_label(provider_name)}."
                if _wants_json(request):
                    return JsonResponse(
                        {
                            "ok": True,
                            "message": message,
                            **_scoring_job_payload(job),
                        },
                        status=202,
                    )
                messages.success(
                    request,
                    message,
                )
                return redirect("practice:scoring_job", pk=job.pk)
        if _wants_json(request):
            return JsonResponse(
                {
                    "ok": False,
                    "error": _form_error_summary(form),
                    "errors": form.errors.get_json_data(escape_html=True),
                },
                status=400,
            )
    else:
        script_id = request.GET.get("script")
        card_id = request.GET.get("card")
        requested_ladder_id = request.GET.get("ladder")
        requested_level = _positive_int(request.GET.get("level"))
        initial_card = None
        initial_script = None
        if card_id:
            initial_card = (
                ImprovementCard.objects.filter(user=request.user)
                .exclude(status=ImprovementCard.STATUS_PAUSED)
                .filter(pk=card_id)
                .first()
            )
        if mode == PracticeRunForm.MODE_QUICK:
            if script_id:
                initial_script = PracticeScript.objects.filter(
                    models.Q(user=request.user) | models.Q(user__isnull=True),
                    pk=script_id,
                    active=True,
                    practice_kind=PracticeScript.KIND_DRILL,
                ).first()
            if initial_script is None and initial_card is not None:
                initial_script = _latest_script_for_card(initial_card)
            selected_ladder = _select_ladder_for_request(
                practice_ladders,
                requested_ladder_id=requested_ladder_id,
                script=initial_script,
            )
            if selected_ladder is not None:
                ladder_steps = _annotate_gate_states(request.user, _ladder_steps(selected_ladder))
                selected_ladder_step = _select_ladder_step(
                    ladder_steps,
                    requested_level=requested_level,
                    script=initial_script,
                )
                if selected_ladder_step is not None and selected_ladder_step.gate_state == "locked":
                    locked_step = selected_ladder_step
                    selected_ladder_step = _highest_non_locked_step(ladder_steps)
                    explicitly_requested_locked = requested_level == locked_step.level or (
                        script_id and str(script_id) == str(locked_step.script_id)
                    )
                    if explicitly_requested_locked:
                        previous_step = _previous_ladder_step(ladder_steps, locked_step)
                        if previous_step is not None:
                            messages.info(
                                request,
                                (
                                    f"Level {locked_step.level} is locked - pass level "
                                    f"{previous_step.level} with {int(previous_step.min_clarity * 100)}% clarity first."
                                ),
                            )
                if selected_ladder_step is not None:
                    initial_script = selected_ladder_step.script
            if initial_script is None and quick_queue:
                initial_card = quick_queue[0].card
                initial_script = quick_queue[0].script
            if initial_script is None:
                initial_script = _first_builtin_drill()
        elif script_id:
            initial_script = PracticeScript.objects.filter(
                models.Q(user=request.user) | models.Q(user__isnull=True),
                pk=script_id,
                active=True,
                practice_kind=PracticeScript.KIND_READING,
            ).first()
        elif initial_card is not None:
            initial_script = _latest_script_for_card(initial_card)
        elif mode == PracticeRunForm.MODE_SCRIPT:
            initial_script = _random_script_for_kind(request.user, PracticeScript.KIND_READING)
        form = PracticeRunForm(
            user=request.user,
            initial_script=initial_script,
            initial_card=initial_card,
            script_kind=script_kind,
            initial={"mode": mode},
        )

    selected_script = None
    script_value = form["script"].value()
    if script_value:
        selected_query = PracticeScript.objects.filter(
            models.Q(user=request.user) | models.Q(user__isnull=True),
            pk=script_value,
        )
        if script_kind:
            selected_query = selected_query.filter(practice_kind=script_kind)
        selected_script = selected_query.first()
    if selected_script is None:
        fallback_scripts = PracticeScript.objects.filter(
            models.Q(user=request.user) | models.Q(user__isnull=True),
            active=True,
        )
        if script_kind:
            fallback_scripts = fallback_scripts.filter(practice_kind=script_kind)
        selected_script = fallback_scripts.first()
    focus_card = None
    card_value = form["card"].value()
    if card_value:
        focus_card = ImprovementCard.objects.filter(user=request.user, pk=card_value).first()
    if mode == PracticeRunForm.MODE_QUICK and selected_ladder is None and selected_script is not None:
        selected_ladder = _select_ladder_for_request(practice_ladders, script=selected_script)
        if selected_ladder is not None:
            ladder_steps = _ladder_steps(selected_ladder)
            selected_ladder_step = _select_ladder_step(ladder_steps, script=selected_script)
    if mode == PracticeRunForm.MODE_QUICK and selected_ladder is not None:
        if not ladder_steps:
            ladder_steps = _ladder_steps(selected_ladder)
        ladder_steps = _annotate_gate_states(request.user, ladder_steps)
        if selected_ladder_step is None:
            selected_ladder_step = _select_ladder_step(ladder_steps, script=selected_script)
    context = {
        "form": form,
        "selected_script": selected_script,
        "focus_card": focus_card,
        "quick_queue": quick_queue,
        "practice_ladders": practice_ladders,
        "selected_ladder": selected_ladder,
        "ladder_steps": ladder_steps,
        "selected_ladder_step": selected_ladder_step,
        "generation_provider_choices": script_generation_provider_choices(request.user),
        "autumn_timer": _autumn_timer_context(
            PracticeSettings.load(request.user),
            selected_script=selected_script,
        ),
    }
    return render(request, "practice/practice_run.html", context)


def session_list(request):
    sessions = PracticeSession.objects.filter(user=request.user)[:80]
    return render(request, "practice/session_list.html", {"sessions": sessions})


def session_detail(request, pk: int):
    session = get_object_or_404(PracticeSession, user=request.user, pk=pk)
    if request.method == "POST":
        action = request.POST.get("action") or "edit_transcript"
        if action == "clear_transcript":
            session.transcript = ""
            session.wer = None
            session.clarity = None
            session.score = None
            session.cer = None
            session.artic_rate = None
            session.pause_ratio = None
            session.filled_pauses = None
            session.avg_conf = None
            session.save(
                update_fields=[
                    "transcript",
                    "wer",
                    "clarity",
                    "score",
                    "cer",
                    "artic_rate",
                    "pause_ratio",
                    "filled_pauses",
                    "avg_conf",
                ]
            )
            SessionError.objects.filter(user=request.user, session_id=session.pk).delete()
            messages.success(request, "Transcript cleared.")
            return redirect("practice:session_detail", pk=session.pk)

        if action in {"save_self_review", "create_cards_from_self_review"}:
            review_form = SelfReviewNotesForm(request.POST, instance=session)
            if review_form.is_valid():
                review_form.save()
                session.refresh_from_db()
                if action == "save_self_review":
                    messages.success(request, "Self-review notes saved.")
                    return redirect("practice:session_detail", pk=session.pk)
                try:
                    created_cards = _create_cards_from_self_review(
                        session,
                        provider=request.POST.get("provider") or None,
                    )
                except Exception as exc:
                    messages.error(request, f"Card generation failed: {exc}")
                    return redirect("practice:session_detail", pk=session.pk)
                if created_cards:
                    messages.success(
                        request,
                        f"Created or updated {len(created_cards)} focus cards from your self-review notes.",
                    )
                    return redirect("practice:cards")
                messages.error(request, "Add a few specific self-review notes before creating cards.")
                return redirect("practice:session_detail", pk=session.pk)
        else:
            review_form = SelfReviewNotesForm(instance=session)

        form = TranscriptEditForm(request.POST, instance=session)
        if form.is_valid():
            transcript = form.cleaned_data.get("transcript") or ""
            result = score_transcript(
                session.script_text,
                TranscriptResult(
                    text=transcript,
                    provider="manual_edit",
                    segments=session_segments(session),
                    raw={},
                ),
            )
            session.transcript = result.transcript
            session.wer = result.wer
            session.clarity = result.clarity
            session.score = result.score
            session.cer = result.cer
            session.artic_rate = result.artic_rate
            session.pause_ratio = result.pause_ratio
            session.filled_pauses = result.filled_pauses
            session.avg_conf = result.avg_conf
            session.save()
            _replace_session_errors(session)
            messages.success(request, "Transcript updated and rescored.")
            return redirect("practice:session_detail", pk=session.pk)
    else:
        form = TranscriptEditForm(instance=session)
        review_form = SelfReviewNotesForm(instance=session)

    highlighted = highlighted_session_text(session)
    mistake_lines = _mistake_lines(session)
    return render(
        request,
        "practice/session_detail.html",
        {
            "session": session,
            "form": form,
            "review_form": review_form,
            "highlighted": highlighted,
            "has_audio": audio_exists(session),
            "mistake_lines": mistake_lines,
            "score_text": _session_score_text(session),
            "generation_provider_choices": script_generation_provider_choices(request.user),
        },
    )


def session_report(request, pk: int):
    session = get_object_or_404(PracticeSession, user=request.user, pk=pk)
    errors = SessionError.objects.filter(user=request.user, session_id=session.pk).order_by("id")[:200]
    lines = [
        f"# SpeechPractice Report: {session.script_name}",
        "",
        f"- Date: {session.timestamp}",
        f"- Score: {_format_metric(session.score)}",
        f"- WER: {_format_metric(session.wer)}",
        f"- CER: {_format_metric(session.cer)}",
        f"- Clarity: {_format_metric(session.clarity)}",
        f"- Articulation rate: {_format_metric(session.artic_rate, suffix=' wpm')}",
        f"- Pause ratio: {_format_metric(session.pause_ratio)}",
        f"- Filled pauses: {_format_metric(session.filled_pauses)}",
        f"- Average confidence: {_format_metric(session.avg_conf)}",
        "",
        "## Target Script",
        "",
        session.script_text or "",
        "",
        "## Transcript",
        "",
        session.transcript or "",
        "",
        "## Self-review Notes",
        "",
        session.self_review_notes or "",
        "",
        "## Mistakes",
        "",
    ]
    if errors:
        for error in errors:
            lines.append(
                f"- {error.error_kind or error.op}: expected `{error.ref_token or ''}`"
                f" heard `{error.hyp_token or ''}`"
            )
    else:
        lines.append("- No stored mistakes.")
    response = HttpResponse("\n".join(lines), content_type="text/markdown; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="speechpractice-session-{session.pk}.md"'
    return response


def _selected_pks(request) -> list[int]:
    pks = []
    for raw in request.POST.getlist("selected"):
        try:
            pks.append(int(raw))
        except (TypeError, ValueError):
            continue
    return pks


@require_POST
def session_bulk_delete(request):
    pks = _selected_pks(request)
    sessions = list(PracticeSession.objects.filter(user=request.user, pk__in=pks))
    if not sessions:
        messages.error(request, "No recordings were selected.")
        return redirect("practice:sessions")
    audio_deleted = 0
    for session in sessions:
        if delete_audio(session.audio_path):
            audio_deleted += 1
    session_ids = [session.pk for session in sessions]
    SessionError.objects.filter(user=request.user, session_id__in=session_ids).delete()
    PracticeReview.objects.filter(user=request.user, legacy_session_id__in=session_ids).delete()
    ScoringJob.objects.filter(user=request.user, legacy_session_id__in=session_ids).update(legacy_session_id=None)
    PracticeSession.objects.filter(user=request.user, pk__in=session_ids).delete()
    suffix = f" and {audio_deleted} audio file{'s' if audio_deleted != 1 else ''}" if audio_deleted else ""
    messages.success(request, f"Deleted {len(session_ids)} recording{'s' if len(session_ids) != 1 else ''}{suffix}.")
    return redirect("practice:sessions")


@require_POST
def session_delete(request, pk: int):
    session = get_object_or_404(PracticeSession, user=request.user, pk=pk)
    script_name = session.script_name
    audio_deleted = delete_audio(session.audio_path)
    SessionError.objects.filter(user=request.user, session_id=session.pk).delete()
    PracticeReview.objects.filter(user=request.user, legacy_session_id=session.pk).delete()
    ScoringJob.objects.filter(user=request.user, legacy_session_id=session.pk).update(legacy_session_id=None)
    session.delete()
    suffix = " and deleted the audio file" if audio_deleted else ""
    messages.success(request, f"Deleted recording entry for {script_name}{suffix}.")
    return redirect("practice:sessions")


def session_audio(request, pk: int):
    session = get_object_or_404(PracticeSession, user=request.user, pk=pk)
    if not stored_audio_exists(session.audio_path):
        raise Http404("Audio file not found.")
    content_type = mimetypes.guess_type(session.audio_path)[0] or "application/octet-stream"
    file_size = audio_size(session.audio_path)
    range_header = request.headers.get("Range", "")
    if range_header.startswith("bytes="):
        start, end = _parse_byte_range(range_header, file_size)
        length = end - start + 1
        with open_audio(session.audio_path, "rb") as audio_file:
            audio_file.seek(start)
            payload = audio_file.read(length)
        response = HttpResponse(payload, status=206, content_type=content_type)
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response["Content-Length"] = str(length)
    else:
        response = FileResponse(
            open_audio(session.audio_path, "rb"),
            as_attachment=False,
            content_type=content_type,
        )
        response["Content-Length"] = str(file_size)
    response["Accept-Ranges"] = "bytes"
    response["Cache-Control"] = "private, max-age=3600"
    return response


def scoring_job_detail(request, pk: int):
    job = get_object_or_404(ScoringJob, user=request.user, pk=pk)
    session = None
    if job.legacy_session_id:
        session = PracticeSession.objects.filter(user=request.user, pk=job.legacy_session_id).first()
    context = {
        "job": job,
        "session": session,
        "unlocked_level": _unlocked_level_for_job_session(job, session),
        **job_status_context(job),
    }
    return render(request, "practice/scoring_job.html", context)


def scoring_job_status(request, pk: int):
    job = get_object_or_404(ScoringJob, user=request.user, pk=pk)
    response = JsonResponse(_scoring_job_payload(job))
    response["Cache-Control"] = "no-store"
    return response


def _scoring_job_payload(job: ScoringJob) -> dict:
    session = (
        PracticeSession.objects.filter(user=job.user, pk=job.legacy_session_id).first()
        if job.legacy_session_id
        else None
    )
    session_url = reverse("practice:session_detail", args=[session.pk]) if session else ""
    payload = {
        "id": job.pk,
        "status": job.status,
        "status_label": job.get_status_display(),
        "is_pending": job.status in {ScoringJob.STATUS_QUEUED, ScoringJob.STATUS_RUNNING},
        "is_done": job.status == ScoringJob.STATUS_SUCCEEDED,
        "is_failed": job.status == ScoringJob.STATUS_FAILED,
        "partial_transcript": job.partial_transcript or "",
        "error_message": job.error_message or "",
        "status_url": reverse("practice:scoring_job_status", args=[job.pk]),
        "job_url": reverse("practice:scoring_job", args=[job.pk]),
        "session_url": session_url,
    }
    if session is not None:
        highlighted = highlighted_session_text(session)
        payload.update(
            {
                "session_id": session.pk,
                "transcript_html": str(highlighted.timed_transcript_html),
                "has_timed_transcript": highlighted.has_timed_transcript,
                "error_count": highlighted.error_count,
                "metrics": _session_metrics_payload(session),
                "score_text": _session_score_text(session),
            }
        )
    return payload


def _session_metrics_payload(session: PracticeSession) -> dict[str, str]:
    return {
        "score": _format_metric(session.score),
        "wer": _format_metric(session.wer),
        "cer": _format_metric(session.cer),
        "clarity": _format_metric(session.clarity),
        "artic_rate": _format_metric(session.artic_rate, suffix=" wpm"),
        "pause_ratio": _format_metric(session.pause_ratio),
        "avg_conf": _format_metric(session.avg_conf),
    }


def _session_score_text(session: PracticeSession) -> str:
    no_scores = not bool(session.transcript)
    score = _format_score_value(session.score, no_scores=no_scores)
    wer = _format_percent_value(session.wer, no_scores=no_scores)
    cer = _format_percent_value(session.cer, no_scores=no_scores)
    clarity = _format_percent_value(session.clarity, no_scores=no_scores)
    rate = _format_wpm_value(session.artic_rate, no_scores=no_scores)
    pauses = _format_percent_value(session.pause_ratio, no_scores=no_scores, places=0)
    conf = _format_percent_value(session.avg_conf, no_scores=no_scores, places=0)
    return (
        f"Score: {score}/5 | WER: {wer} | CER: {cer} | "
        f"Clarity: {clarity} | Rate: {rate} | Pauses: {pauses} | Conf: {conf}"
    )


def _format_score_value(value: float | None, no_scores: bool = False) -> str:
    if value is None or no_scores:
        return "-"
    return f"{float(value):.2f}"


def _format_percent_value(
    value: float | None,
    no_scores: bool = False,
    places: int = 2,
) -> str:
    if value is None or no_scores:
        return "-"
    return f"{float(value):.{places}%}"


def _format_wpm_value(value: float | None, no_scores: bool = False) -> str:
    if value is None or no_scores:
        return "- wpm"
    return f"{float(value):.0f} wpm"


def _wants_json(request) -> bool:
    return (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )


def _form_error_summary(form) -> str:
    messages_list: list[str] = []
    for field_name, errors in form.errors.items():
        label = ""
        if field_name != "__all__" and field_name in form.fields:
            label = form.fields[field_name].label or field_name.replace("_", " ").title()
        for error in errors:
            messages_list.append(f"{label}: {error}" if label else str(error))
    return " ".join(messages_list) or "Please fix the highlighted fields."


@require_POST
def retry_scoring_job(request, pk: int):
    job = get_object_or_404(ScoringJob, user=request.user, pk=pk)
    if job.status != ScoringJob.STATUS_FAILED:
        messages.error(request, "Only failed scoring jobs can be retried.")
        return redirect("practice:scoring_job", pk=job.pk)
    job.status = ScoringJob.STATUS_QUEUED
    job.error_message = ""
    job.started_at = None
    job.finished_at = None
    job.save(update_fields=["status", "error_message", "started_at", "finished_at", "updated_at"])
    enqueue_scoring_job(job)
    messages.success(request, "Scoring job requeued.")
    return redirect("practice:scoring_job", pk=job.pk)


def script_list(request):
    kind_filter = (request.GET.get("kind") or PracticeScript.KIND_READING).strip()
    valid_kinds = {value for value, _label in PracticeScript.KIND_CHOICES}
    if kind_filter not in valid_kinds:
        kind_filter = PracticeScript.KIND_READING
    visible_scripts = PracticeScript.objects.filter(
        models.Q(user=request.user) | models.Q(user__isnull=True)
    )
    scripts = visible_scripts.filter(practice_kind=kind_filter)
    source_filter = (request.GET.get("source") or "").strip()
    total_count = scripts.count()
    if source_filter:
        scripts = scripts.filter(source=source_filter)
    groups = _script_source_groups(scripts)
    kind_counts = {
        row["practice_kind"]: row["count"]
        for row in visible_scripts.values("practice_kind").annotate(count=models.Count("id"))
    }
    source_counts = {
        row["source"]: row["count"]
        for row in visible_scripts.filter(practice_kind=kind_filter)
        .values("source")
        .annotate(count=models.Count("id"))
    }
    kind_tabs = [
        {
            "value": value,
            "label": "Reading scripts" if value == PracticeScript.KIND_READING else "Ladders / Drills",
            "count": kind_counts.get(value, 0),
            "active": kind_filter == value,
        }
        for value, _label in PracticeScript.KIND_CHOICES
    ]
    source_tabs = [
        {
            "value": value,
            "label": label,
            "count": source_counts.get(value, 0),
            "active": source_filter == value,
        }
        for value, label in PracticeScript.SOURCE_CHOICES
    ]
    return render(
        request,
        "practice/script_list.html",
        {
            "scripts": scripts,
            "kind_filter": kind_filter,
            "kind_tabs": kind_tabs,
            "source_filter": source_filter,
            "total_count": total_count,
            "source_groups": groups,
            "source_tabs": source_tabs,
            "practice_ladders": _practice_ladders(request.user) if kind_filter == PracticeScript.KIND_DRILL else [],
            "ladder_candidate_cards": _ladder_candidate_cards(request.user) if kind_filter == PracticeScript.KIND_DRILL else [],
            "default_ladder_card_ids": _default_ladder_card_ids(request.user) if kind_filter == PracticeScript.KIND_DRILL else set(),
            "generation_provider_choices": script_generation_provider_choices(request.user),
        },
    )


def script_create(request):
    if request.method == "POST":
        form = PracticeScriptForm(request.POST)
        if form.is_valid():
            script = form.save(commit=False)
            script.user = request.user
            script.save()
            form.save_m2m()
            messages.success(request, "Practice script saved.")
            return redirect("practice:scripts")
    else:
        form = PracticeScriptForm(
            initial={
                "practice_kind": PracticeScript.KIND_READING,
                "source": PracticeScript.SOURCE_USER,
                "difficulty": 1,
            }
        )
    return render(request, "practice/script_form.html", {"form": form})


def script_edit(request, pk: int):
    script = get_object_or_404(
        PracticeScript.objects.filter(
            models.Q(user=request.user)
            | (models.Q(user__isnull=True) & ~models.Q(source=PracticeScript.SOURCE_BUILTIN))
        ),
        pk=pk,
    )
    if script.user_id is None:
        script.user = request.user
        script.save(update_fields=["user", "updated_at"])
    if request.method == "POST":
        form = PracticeScriptForm(request.POST, instance=script)
        if form.is_valid():
            form.save()
            messages.success(request, "Practice script updated.")
            return redirect("practice:scripts")
    else:
        form = PracticeScriptForm(instance=script)
    return render(
        request,
        "practice/script_form.html",
        {
            "form": form,
            "script": script,
            "is_edit": True,
        },
    )


def _script_list_redirect(request):
    kind = (request.POST.get("kind") or "").strip()
    valid_kinds = {value for value, _label in PracticeScript.KIND_CHOICES}
    if kind not in valid_kinds:
        kind = PracticeScript.KIND_READING
    source = (request.POST.get("source") or "").strip()
    valid_sources = {value for value, _label in PracticeScript.SOURCE_CHOICES}
    url = f"{reverse('practice:scripts')}?kind={kind}"
    if source in valid_sources:
        url += f"&source={source}"
    return redirect(url)


@require_POST
def script_bulk_delete(request):
    pks = _selected_pks(request)
    scripts = PracticeScript.objects.filter(
        models.Q(user=request.user)
        | (models.Q(user__isnull=True) & ~models.Q(source=PracticeScript.SOURCE_BUILTIN)),
        pk__in=pks,
    )
    deleted = scripts.count()
    if not deleted:
        messages.error(request, "No scripts were selected.")
        return _script_list_redirect(request)
    scripts.delete()
    messages.success(request, f"Deleted {deleted} script{'s' if deleted != 1 else ''}.")
    return _script_list_redirect(request)


def script_delete(request, pk: int):
    script = get_object_or_404(
        PracticeScript.objects.filter(
            models.Q(user=request.user)
            | (models.Q(user__isnull=True) & ~models.Q(source=PracticeScript.SOURCE_BUILTIN))
        ),
        pk=pk,
    )
    if request.method == "POST":
        title = script.title
        script.delete()
        messages.success(request, f"Deleted script: {title}.")
        return redirect("practice:scripts")
    return render(request, "practice/script_confirm_delete.html", {"script": script})


def script_preview(request, pk: int):
    script = get_object_or_404(
        PracticeScript.objects.filter(models.Q(user=request.user) | models.Q(user__isnull=True)),
        pk=pk,
        active=True,
    )
    return JsonResponse(
        {
            "id": script.pk,
            "title": script.title,
            "author": script.author,
            "body": script.body,
            "word_count": script.word_count,
            "practice_kind": script.get_practice_kind_display(),
            "source": script.get_source_display(),
            "difficulty": script.difficulty,
            "tags": script.tags or [],
        }
    )


def script_import(request):
    results = None
    if request.method == "POST":
        form = BulkScriptImportForm(request.POST, request.FILES)
        if form.is_valid():
            items = []
            for uploaded in form.cleaned_data["files"]:
                items.extend(
                    parse_script_upload(
                        name=uploaded.name,
                        content=uploaded.read(),
                        default_author=form.cleaned_data.get("default_author", ""),
                    )
                )
            results = import_script_items(
                items,
                user=request.user,
                source=PracticeScript.SOURCE_UPLOADED,
                extra_tags=form.tags(),
                replace=form.cleaned_data.get("replace", True),
            )
            messages.success(
                request,
                f"Import complete: {results.created} created, {results.updated} updated, {results.skipped} skipped.",
            )
    else:
        form = BulkScriptImportForm()
    return render(
        request,
        "practice/script_import.html",
        {
            "form": form,
            "results": results,
        },
    )


def card_list(request):
    if request.method == "POST" and request.POST.get("refresh") == "1":
        start_dt, end_dt, error = _card_refresh_range(request.POST.get("start"), request.POST.get("end"))
        if error:
            messages.error(request, error)
            return redirect("practice:cards")
        created = refresh_improvement_cards(
            user=request.user,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        messages.success(
            request,
            f"Cards refreshed from {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}. {created} new focus areas created.",
        )
        return redirect("practice:cards")
    cards = list(ImprovementCard.objects.filter(user=request.user))
    grouped_cards = []
    for kind, label in ImprovementCard.KIND_CHOICES:
        group_cards = [card for card in cards if card.kind == kind]
        if not group_cards:
            continue
        grouped_cards.append(
            {
                "kind": kind,
                "label": label,
                "cards": group_cards,
                "count": len(group_cards),
                "due_count": sum(1 for card in group_cards if card.due_at <= timezone.now()),
            }
        )
    today = timezone.localdate()
    return render(
        request,
        "practice/card_list.html",
        {
            "cards": cards,
            "grouped_cards": grouped_cards,
            "total_count": len(cards),
            "refresh_default_start": today - timedelta(days=30),
            "refresh_default_end": today,
        },
    )


def _create_cards_from_self_review(
    session: PracticeSession,
    *,
    provider: str | None = None,
) -> list[ImprovementCard]:
    notes = (session.self_review_notes or "").strip()
    if not notes:
        return []
    draft = generate_cards_from_self_review(session, notes, provider_name=provider)
    cards = []
    for card_draft in draft.cards:
        if not card_draft.target_key:
            continue
        stats = {
            **card_draft.stats,
            "source_provider": draft.provider,
            "source_auth": draft.auth_source,
        }
        card, _created = ImprovementCard.objects.update_or_create(
            user=session.user,
            kind=card_draft.kind,
            target_key=card_draft.target_key,
            defaults={
                "title": card_draft.title,
                "prompt": card_draft.prompt,
                "stats": stats,
                "status": ImprovementCard.STATUS_LEARNING,
                "due_at": timezone.now(),
            },
        )
        cards.append(card)
    return cards


def _card_refresh_range(
    start_raw: str | None,
    end_raw: str | None,
) -> tuple[datetime | None, datetime | None, str]:
    if not start_raw or not end_raw:
        return None, None, "Choose a start and end date for card refresh."
    try:
        start_dt = datetime.strptime(start_raw, "%Y-%m-%d")
        end_dt = datetime.strptime(end_raw, "%Y-%m-%d").replace(
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
        )
    except ValueError:
        return None, None, "Use valid start and end dates for card refresh."
    if start_dt > end_dt:
        return None, None, "Start date must be before end date."
    return start_dt, end_dt, ""


def card_detail(request, pk: int):
    card = get_object_or_404(ImprovementCard, user=request.user, pk=pk)
    reviews = card.reviews.all()[:8]
    generated_scripts = PracticeScript.objects.filter(
        models.Q(user=request.user) | models.Q(user__isnull=True),
        generation_records__card=card,
        practice_kind=PracticeScript.KIND_DRILL,
    ).distinct()[:8]
    evidence_rows = _card_evidence_rows(card)
    story_context = _card_story(card)
    return render(
        request,
        "practice/card_detail.html",
        {
            "card": card,
            "reviews": reviews,
            "generated_scripts": generated_scripts,
            "evidence_rows": evidence_rows,
            "generation_provider_choices": script_generation_provider_choices(request.user),
            **story_context,
        },
    )


def _card_story(card: ImprovementCard) -> dict:
    reviews = list(PracticeReview.objects.filter(user=card.user, card=card).order_by("reviewed_at", "pk"))
    generations = list(
        GeneratedPracticeScript.objects.filter(
            user=card.user,
            card=card,
            script__isnull=False,
        )
        .select_related("script")
        .order_by("created_at", "pk")
    )
    ladders = list(
        PracticeLadder.objects.filter(
            models.Q(user=card.user) | models.Q(user__isnull=True),
            active=True,
            cards=card,
        )
        .distinct()
        .order_by("created_at", "pk")
    )

    source_label = str((card.stats or {}).get("source_window_label") or "").strip()
    events = [
        {
            "kind": "created",
            "at": card.created_at,
            "title": "Spotted",
            "detail": f"Flagged from {source_label}" if source_label else "Flagged for practice",
            "url": None,
        }
    ]

    practice_url = reverse("practice:practice")
    events.extend(_card_story_drill_events(generations, practice_url, card.pk))
    events.extend(
        {
            "kind": "ladder",
            "at": ladder.created_at,
            "title": "Added to ladder",
            "detail": ladder.title,
            "url": f"{practice_url}?mode=quick&ladder={ladder.pk}",
        }
        for ladder in ladders
    )
    events.extend(_card_story_review_events(reviews))

    if card.status == ImprovementCard.STATUS_MASTERED:
        events.append(
            {
                "kind": "mastered",
                "at": card.last_reviewed_at or card.updated_at,
                "title": "Mastered",
                "detail": f"Mastery {float(card.mastery or 0.0):.2f}",
                "url": None,
            }
        )

    events.sort(key=lambda event: (event["at"], event["kind"], event["title"]))
    return {
        "story_events": events,
        "mastery_curve": _card_mastery_curve(card, reviews),
        "review_stats": _card_review_stats(reviews),
    }


def _card_story_drill_events(generations: list[GeneratedPracticeScript], practice_url: str, card_pk: int) -> list[dict]:
    if len(generations) > 6:
        skipped_count = len(generations) - 5
        visible = [
            *generations[:2],
            {
                "synthetic": True,
                "at": generations[2].created_at,
                "count": skipped_count,
            },
            *generations[-3:],
        ]
    else:
        visible = generations

    events = []
    for generation in visible:
        if isinstance(generation, dict):
            events.append(
                {
                    "kind": "drill",
                    "at": generation["at"],
                    "title": "More drills",
                    "detail": f"{generation['count']} more drills generated",
                    "url": None,
                }
            )
            continue
        events.append(
            {
                "kind": "drill",
                "at": generation.created_at,
                "title": "Drill generated",
                "detail": generation.script.title,
                "url": f"{practice_url}?mode=quick&script={generation.script_id}&card={card_pk}",
            }
        )
    return events


def _card_story_review_events(reviews: list[PracticeReview]) -> list[dict]:
    if len(reviews) > 8:
        skipped_count = len(reviews) - 6
        visible = [
            *reviews[:2],
            {
                "synthetic": True,
                "at": reviews[2].reviewed_at,
                "count": skipped_count,
            },
            *reviews[-4:],
        ]
    else:
        visible = reviews

    events = []
    for review in visible:
        if isinstance(review, dict):
            events.append(
                {
                    "kind": "review",
                    "at": review["at"],
                    "title": "More reviews",
                    "detail": f"{review['count']} more reviews",
                    "url": None,
                }
            )
            continue
        parts = []
        if review.quality is not None:
            parts.append(f"Quality {review.quality:.2f}")
        if review.error_rate is not None:
            parts.append(f"target error rate {int(round(review.error_rate * 100))}%")
        events.append(
            {
                "kind": "review",
                "at": review.reviewed_at,
                "title": "Reviewed",
                "detail": " - ".join(parts) if parts else "Review logged",
                "url": None,
            }
        )
    return events


def _card_mastery_curve(card: ImprovementCard, reviews: list[PracticeReview]) -> dict | None:
    points = [review for review in reviews if review.mastery_after is not None]
    if len(points) < 2:
        return None

    span = len(points) - 1
    coords = []
    for index, review in enumerate(points):
        mastery = max(0.0, min(1.0, float(review.mastery_after or 0.0)))
        x = round(20 + (560 * index / span), 1)
        y = round(140 - (120 * mastery), 1)
        coords.append((x, y))

    crosses_year = points[0].reviewed_at.year != points[-1].reviewed_at.year
    date_format = "%b %d, %Y" if crosses_year else "%b %d"
    return {
        "viewbox": "0 0 600 160",
        "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in coords),
        "last_x": coords[-1][0],
        "last_y": coords[-1][1],
        "start_label": points[0].reviewed_at.strftime(date_format),
        "end_label": points[-1].reviewed_at.strftime(date_format),
        "current_mastery": float(card.mastery or 0.0),
    }


def _card_review_stats(reviews: list[PracticeReview]) -> dict:
    evidence_reviews = [review for review in reviews if review.evidence is not None]
    evidence_error_rates = [review.error_rate for review in evidence_reviews if review.error_rate is not None]
    return {
        "count": len(reviews),
        "evidence_count": len(evidence_reviews),
        "first_at": reviews[0].reviewed_at if reviews else None,
        "last_at": reviews[-1].reviewed_at if reviews else None,
        "first_error_rate": evidence_error_rates[0] if evidence_error_rates else None,
        "latest_error_rate": evidence_error_rates[-1] if evidence_error_rates else None,
    }


@require_POST
def card_bulk_delete(request):
    pks = _selected_pks(request)
    cards = ImprovementCard.objects.filter(user=request.user, pk__in=pks)
    deleted = cards.count()
    if not deleted:
        messages.error(request, "No cards were selected.")
        return redirect("practice:cards")
    cards.delete()
    messages.success(
        request,
        f"Deleted {deleted} card{'s' if deleted != 1 else ''}. Existing drills and history were kept.",
    )
    return redirect("practice:cards")


@require_POST
def card_delete(request, pk: int):
    card = get_object_or_404(ImprovementCard, user=request.user, pk=pk)
    title = card.display_title
    card.delete()
    messages.success(request, f"Deleted card: {title}. Existing drills and history were kept.")
    return redirect("practice:cards")


@require_POST
def generate_script_for_card(request, pk: int):
    card = get_object_or_404(ImprovementCard, user=request.user, pk=pk)
    provider = request.POST.get("provider") or None
    try:
        draft = generate_script_draft(card, provider_name=provider)
    except Exception as exc:
        messages.error(request, f"Script generation failed: {exc}")
        return redirect("practice:card_detail", pk=card.pk)
    script = PracticeScript.objects.create(
        user=request.user,
        title=draft.title,
        body=draft.body,
        practice_kind=PracticeScript.KIND_DRILL,
        source=PracticeScript.SOURCE_GENERATED,
        source_ref=f"card:{card.pk}",
        tags=[card.kind, card.target_key],
        target_patterns=[{"kind": card.kind, "target": card.target_key}],
        difficulty=2,
    )
    GeneratedPracticeScript.objects.create(
        user=request.user,
        card=card,
        script=script,
        model_provider=draft.provider,
        auth_source=draft.auth_source,
        prompt_snapshot=draft.prompt_snapshot,
    )
    messages.success(
        request,
        f"Generated a starter practice script for this card using {_generation_source_label(draft.provider, draft.auth_source)}.",
    )
    if request.POST.get("next") == "practice":
        return redirect(f"{reverse('practice:practice')}?mode=quick&script={script.pk}&card={card.pk}")
    return redirect("practice:scripts")


@require_POST
def generate_practice_ladder(request):
    provider = request.POST.get("provider") or None
    theme = (request.POST.get("theme") or "").strip()[:512]
    cards = _cards_for_ladder_generation(request.user, request.POST.getlist("cards"))
    if not cards:
        messages.error(request, "Add or refresh improvement cards before generating a practice ladder.")
        return redirect(f"{reverse('practice:practice')}?mode=quick")

    try:
        draft = generate_ladder_draft(cards, theme=theme, provider_name=provider)
    except Exception as exc:
        messages.error(request, f"Practice ladder generation failed: {exc}")
        return redirect(f"{reverse('practice:practice')}?mode=quick")

    ladder = PracticeLadder.objects.create(
        user=request.user,
        title=draft.title,
        theme=draft.theme or theme,
        source=PracticeLadder.SOURCE_GENERATED,
        source_ref="cards:" + ",".join(str(card.pk) for card in cards),
        model_provider=draft.provider,
        auth_source=draft.auth_source,
        prompt_snapshot=draft.prompt_snapshot,
    )
    ladder.cards.set(cards)
    target_patterns = [
        {"kind": card.kind, "target": card.target_key}
        for card in cards
        if card.target_key
    ]
    created_steps = []
    used_levels: set[int] = set()
    for level_draft in sorted(draft.levels, key=lambda item: item.level)[:5]:
        requested_level = max(1, min(5, int(level_draft.level or len(created_steps) + 1)))
        level = requested_level if requested_level not in used_levels else 0
        if not level:
            level = next(candidate for candidate in range(1, 6) if candidate not in used_levels)
        used_levels.add(level)
        tags = _unique_tags(
            ["generated-ladder", f"level-{level}", *level_draft.focus]
            + [card.target_key for card in cards if card.target_key]
        )
        script = PracticeScript.objects.create(
            user=request.user,
            title=f"{ladder.title}: {level_draft.title}",
            body=level_draft.body,
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
            source_ref=f"ladder:{ladder.pk}:level:{level}",
            tags=tags,
            target_patterns=target_patterns,
            difficulty=level,
        )
        step = PracticeLadderStep.objects.create(
            ladder=ladder,
            script=script,
            level=level,
            title=level_draft.title,
            focus=list(level_draft.focus),
            min_clarity=LADDER_MIN_CLARITY_BY_LEVEL.get(level, 0.95),
        )
        created_steps.append(step)
        for card in cards:
            GeneratedPracticeScript.objects.create(
                user=request.user,
                card=card,
                script=script,
                model_provider=draft.provider,
                auth_source=draft.auth_source,
                prompt_snapshot=draft.prompt_snapshot,
            )

    if not created_steps:
        ladder.delete()
        messages.error(request, "Practice ladder generation did not return any usable levels.")
        return redirect(f"{reverse('practice:practice')}?mode=quick")

    first_step = sorted(created_steps, key=lambda item: item.level)[0]
    messages.success(
        request,
        f"Generated a five-step practice ladder using {_generation_source_label(draft.provider, draft.auth_source)}.",
    )
    if request.POST.get("next") == "scripts":
        return redirect(f"{reverse('practice:scripts')}?kind=drill")
    return redirect(
        f"{reverse('practice:practice')}?mode=quick&ladder={ladder.pk}&level={first_step.level}&script={first_step.script_id}"
    )


def _delete_ladder_with_scripts(request, ladder) -> int:
    """Delete a non-builtin ladder plus its generated drill scripts; returns script count."""
    if ladder.user_id is None:
        ladder.user = request.user
        ladder.save(update_fields=["user", "updated_at"])
    script_ids = list(
        ladder.steps.filter(
            script__source=PracticeScript.SOURCE_GENERATED,
            script__source_ref__startswith=f"ladder:{ladder.pk}:",
        ).values_list("script_id", flat=True)
    )
    ladder.delete()
    scripts_to_delete = PracticeScript.objects.filter(
        models.Q(user=request.user) | models.Q(user__isnull=True),
        pk__in=script_ids,
    )
    deleted_scripts = scripts_to_delete.count()
    if deleted_scripts:
        scripts_to_delete.delete()
    return deleted_scripts


@require_POST
def delete_practice_ladder(request, pk: int):
    ladder = get_object_or_404(
        PracticeLadder.objects.filter(models.Q(user=request.user) | models.Q(user__isnull=True)),
        pk=pk,
    )
    if ladder.source == PracticeLadder.SOURCE_BUILTIN:
        messages.error(request, "Built-in ladders cannot be deleted.")
        return redirect(f"{reverse('practice:scripts')}?kind=drill")

    title = ladder.title
    deleted_scripts = _delete_ladder_with_scripts(request, ladder)
    suffix = f" and {deleted_scripts} generated drill scripts" if deleted_scripts else ""
    messages.success(request, f"Deleted ladder: {title}{suffix}.")
    return redirect(f"{reverse('practice:scripts')}?kind=drill")


@require_POST
def ladder_bulk_delete(request):
    pks = _selected_pks(request)
    ladders = list(
        PracticeLadder.objects.filter(
            models.Q(user=request.user) | models.Q(user__isnull=True),
            pk__in=pks,
        ).exclude(source=PracticeLadder.SOURCE_BUILTIN)
    )
    if not ladders:
        messages.error(request, "No ladders were selected.")
        return redirect(f"{reverse('practice:scripts')}?kind=drill")
    deleted_scripts = 0
    for ladder in ladders:
        deleted_scripts += _delete_ladder_with_scripts(request, ladder)
    suffix = f" and {deleted_scripts} generated drill scripts" if deleted_scripts else ""
    messages.success(request, f"Deleted {len(ladders)} ladder{'s' if len(ladders) != 1 else ''}{suffix}.")
    return redirect(f"{reverse('practice:scripts')}?kind=drill")


def account_settings(request):
    settings_obj = PracticeSettings.load(request.user)
    if request.method == "POST":
        if "connect_autumn" in request.POST:
            _connect_autumn(request, settings_obj)
            return redirect("practice:account")

        if "disconnect_autumn" in request.POST:
            settings_obj.set_secret("autumn_token", None)
            settings_obj.autumn_active_session_id = None
            settings_obj.save()
            messages.success(request, "Autumn disconnected.")
            return redirect("practice:account")

        if "start_codex_login" in request.POST:
            try:
                device_code = start_device_code_login()
            except CodexAuthError as exc:
                messages.error(request, f"Could not start Codex login: {exc}")
            else:
                request.session["codex_device_code"] = device_code.as_session_dict()
                messages.success(request, "Codex login started. Enter the code below, then complete login.")
            return redirect("practice:account")

        if "complete_codex_login" in request.POST:
            device_code = request.session.get("codex_device_code")
            if not device_code:
                messages.error(request, "Start Codex login first.")
                return redirect("practice:account")
            try:
                bundle = poll_device_code_login(device_code)
            except CodexDevicePending:
                messages.info(request, "Still waiting for OpenAI authorization.")
            except CodexAuthError as exc:
                messages.error(request, f"Could not complete Codex login: {exc}")
            else:
                settings_obj.set_secret("codex_token_bundle", serialize_token_bundle(bundle))
                settings_obj.save()
                request.session.pop("codex_device_code", None)
                messages.success(request, "Codex login connected.")
            return redirect("practice:account")

        if "disconnect_codex" in request.POST:
            settings_obj.set_secret("codex_token_bundle", None)
            settings_obj.save()
            request.session.pop("codex_device_code", None)
            messages.success(request, "Codex login disconnected.")
            return redirect("practice:account")

        if "clear_whisper_cache" in request.POST:
            clear_local_whisper_cache()
            messages.success(request, "Local Whisper model cache cleared.")
            return redirect("practice:account")

        if "start_autumn_timer" in request.POST:
            _start_autumn_timer(request, settings_obj, request.POST.get("autumn_note"))
            return redirect("practice:account")

        if "stop_autumn_timer" in request.POST:
            _stop_autumn_timer(request, settings_obj, request.POST.get("autumn_note"))
            return redirect("practice:account")

        old_whisper = _whisper_settings_snapshot(settings_obj)
        form = AccountSettingsForm(
            request.POST,
            instance=settings_obj,
            **_autumn_form_options(
                settings_obj,
                project=request.POST.get("autumn_project"),
            ),
        )
        if form.is_valid():
            form.save()
            if old_whisper != _whisper_settings_snapshot(settings_obj):
                clear_local_whisper_cache()
            messages.success(request, "Account settings saved.")
            return redirect("practice:account")
    else:
        form = AccountSettingsForm(instance=settings_obj, **_autumn_form_options(settings_obj))

    bundle = deserialize_token_bundle(settings_obj.get_secret("codex_token_bundle"))
    context = {
        "form": form,
        "settings_obj": settings_obj,
        "have_keys": {
            "openai": settings_obj.has_secret("openai_api_key"),
            "anthropic": settings_obj.has_secret("anthropic_api_key"),
            "autumn": settings_obj.has_secret("autumn_token"),
            "codex": settings_obj.has_secret("codex_token_bundle"),
        },
        "codex_device_code": request.session.get("codex_device_code"),
        "codex_summary": token_bundle_summary(bundle),
        "autumn_configured": settings_obj.has_secret("autumn_token") and bool(settings_obj.autumn_project),
        "autumn_active": bool(settings_obj.autumn_active_session_id),
        "autumn_note": _latest_autumn_note(request.user),
    }
    return render(request, "practice/account.html", context)


def autumn_subprojects(request):
    settings_obj = PracticeSettings.load(request.user)
    project = (request.GET.get("project") or settings_obj.autumn_project or "").strip()
    if not project:
        return JsonResponse({"ok": False, "error": "Choose an Autumn project first."}, status=400)
    try:
        client = _autumn_token_client(settings_obj)
        subprojects = client.list_subprojects(project)
    except AutumnError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    return JsonResponse({"ok": True, "subprojects": subprojects})


@require_POST
def autumn_timer(request):
    settings_obj = PracticeSettings.load(request.user)
    next_url = _safe_next_url(request, request.POST.get("next")) or reverse("practice:practice")
    if "stop_autumn_timer" in request.POST:
        result = _stop_autumn_timer(
            request,
            settings_obj,
            request.POST.get("autumn_note"),
            notify=not _wants_json(request),
        )
    elif "start_autumn_timer" in request.POST:
        result = _start_autumn_timer(
            request,
            settings_obj,
            request.POST.get("autumn_note"),
            notify=not _wants_json(request),
        )
    else:
        result = {
            "ok": False,
            "active": bool(settings_obj.autumn_active_session_id),
            "message": "Choose whether to start or stop Autumn.",
        }
        if not _wants_json(request):
            messages.error(request, result["message"])
    if _wants_json(request):
        status = 200 if result["ok"] else 400
        return JsonResponse(result, status=status)
    return redirect(next_url)


def _latest_script_for_card(card: ImprovementCard) -> PracticeScript | None:
    generated = (
        GeneratedPracticeScript.objects.select_related("script")
        .filter(
            user=card.user,
            card=card,
            script__active=True,
            script__practice_kind=PracticeScript.KIND_DRILL,
        )
        .order_by("-created_at")
        .first()
    )
    return generated.script if generated and generated.script else None


def _practice_ladders(user) -> list[PracticeLadder]:
    return list(
        PracticeLadder.objects.filter(models.Q(user=user) | models.Q(user__isnull=True), active=True)
        .prefetch_related("cards", "steps__script")
        .order_by("source", "-created_at", "title")
    )


def _select_ladder_for_request(
    ladders: list[PracticeLadder],
    requested_ladder_id: str | None = None,
    script: PracticeScript | None = None,
) -> PracticeLadder | None:
    if requested_ladder_id:
        for ladder in ladders:
            if str(ladder.pk) == str(requested_ladder_id):
                return ladder
    if script is not None:
        ladder_ids = [ladder.pk for ladder in ladders]
        step = (
            PracticeLadderStep.objects.select_related("ladder")
            .filter(script=script, ladder__active=True, ladder_id__in=ladder_ids)
            .order_by("ladder__source", "-ladder__created_at", "level")
            .first()
        )
        if step is not None:
            return step.ladder
    return ladders[0] if ladders else None


def _ladder_steps(ladder: PracticeLadder) -> list[PracticeLadderStep]:
    return list(
        ladder.steps.select_related("script")
        .filter(script__active=True)
        .order_by("level")
    )


def _annotate_gate_states(user, steps: list[PracticeLadderStep]) -> list[PracticeLadderStep]:
    progress_by_step = {
        progress.step_id: progress
        for progress in LadderStepProgress.objects.filter(
            user=user,
            step_id__in=[step.pk for step in steps],
        )
    }
    previous_passed = False
    for index, step in enumerate(steps):
        progress = progress_by_step.get(step.pk)
        passed = bool(progress and progress.passed_at)
        unlocked = index == 0 or previous_passed
        step.gate_state = "passed" if passed else ("open" if unlocked else "locked")
        step.best_clarity = (
            float(progress.best_clarity)
            if progress is not None and progress.attempts > 0
            else None
        )
        previous_passed = passed
    return steps


def _highest_non_locked_step(steps: list[PracticeLadderStep]) -> PracticeLadderStep | None:
    available = [step for step in steps if getattr(step, "gate_state", None) != "locked"]
    return available[-1] if available else None


def _previous_ladder_step(
    steps: list[PracticeLadderStep],
    step: PracticeLadderStep,
) -> PracticeLadderStep | None:
    for index, candidate in enumerate(steps):
        if candidate.pk == step.pk:
            return steps[index - 1] if index > 0 else None
    return None


def _locked_ladder_step_for_script(user, script: PracticeScript) -> PracticeLadderStep | None:
    ladder_ids = list(
        PracticeLadderStep.objects.filter(
            script=script,
            ladder__active=True,
        )
        .filter(models.Q(ladder__user=user) | models.Q(ladder__user__isnull=True))
        .values_list("ladder_id", flat=True)
        .distinct()
    )
    for ladder in PracticeLadder.objects.filter(pk__in=ladder_ids).order_by("source", "-created_at", "title"):
        steps = _annotate_gate_states(user, _ladder_steps(ladder))
        for step in steps:
            if step.script_id == script.pk and step.gate_state == "locked":
                return step
    return None


def _unlocked_level_for_job_session(job: ScoringJob, session: PracticeSession | None) -> int | None:
    if job.status != ScoringJob.STATUS_SUCCEEDED or session is None or job.script_id is None:
        return None
    steps = list(
        job.script.ladder_steps.select_related("ladder")
        .filter(ladder__active=True)
        .order_by("ladder_id", "level")
    )
    for step in steps:
        progress = LadderStepProgress.objects.filter(user=job.user, step=step).first()
        if not progress or not progress.passed_at or progress.best_session_id != session.pk:
            continue
        next_step = (
            PracticeLadderStep.objects.filter(
                ladder=step.ladder,
                level=step.level + 1,
                script__active=True,
            )
            .order_by("level")
            .first()
        )
        if next_step is not None:
            return next_step.level
    return None


def _select_ladder_step(
    steps: list[PracticeLadderStep],
    requested_level: int | None = None,
    script: PracticeScript | None = None,
) -> PracticeLadderStep | None:
    if script is not None:
        for step in steps:
            if step.script_id == script.pk:
                return step
    if requested_level is not None:
        for step in steps:
            if step.level == requested_level:
                return step
    return steps[0] if steps else None


def _first_builtin_drill() -> PracticeScript | None:
    return (
        PracticeScript.objects.filter(
            active=True,
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_BUILTIN,
            source_ref__startswith="builtin:speech-drills:",
        )
        .order_by("difficulty", "title")
        .first()
    )


def _random_script_for_kind(user, script_kind: str) -> PracticeScript | None:
    scripts = PracticeScript.objects.filter(
        models.Q(user=user) | models.Q(user__isnull=True),
        active=True,
        practice_kind=script_kind,
    ).order_by("pk")
    count = scripts.count()
    if count <= 0:
        return None
    return scripts[random.randrange(count)]


def _cards_for_ladder_generation(user, card_ids: list[str]) -> list[ImprovementCard]:
    if card_ids:
        ids = [_positive_int(raw) for raw in card_ids]
        id_order = [card_id for card_id in ids if card_id]
        cards = list(
            ImprovementCard.objects.filter(user=user)
            .exclude(status=ImprovementCard.STATUS_PAUSED)
            .filter(
                pk__in=id_order,
            )
        )
        cards_by_id = {card.pk: card for card in cards}
        ordered = [cards_by_id[card_id] for card_id in id_order if card_id in cards_by_id]
        if ordered:
            return ordered
    return [item.card for item in today_queue(user, limit=4)]


def _ladder_candidate_cards(user) -> list[ImprovementCard]:
    return list(
        ImprovementCard.objects.filter(user=user)
        .exclude(status=ImprovementCard.STATUS_PAUSED)
        .order_by("due_at", "mastery", "title")[:120]
    )


def _default_ladder_card_ids(user) -> set[int]:
    return {item.card.pk for item in today_queue(user, limit=4)}


def _positive_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _unique_tags(tags: list[str]) -> list[str]:
    seen = set()
    result = []
    for tag in tags:
        clean = str(tag or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result[:12]


def _generation_source_label(provider: str, auth_source: str) -> str:
    if auth_source == "codex":
        return f"{provider} through Codex auth"
    if auth_source == "api_key":
        return f"{provider} with an API key"
    if auth_source == "api_key_fallback":
        return f"{provider} with an API key after Codex auth failed"
    if auth_source == "local":
        return provider
    return provider


def _script_kind_for_mode(mode: str | None) -> str | None:
    if mode == PracticeRunForm.MODE_QUICK:
        return PracticeScript.KIND_DRILL
    if mode == PracticeRunForm.MODE_FREE:
        return None
    return PracticeScript.KIND_READING


def _builtin_drills(limit: int = 5):
    return list(
        PracticeScript.objects.filter(
            active=True,
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_BUILTIN,
            source_ref__startswith="builtin:speech-drills:",
        ).order_by("difficulty", "title")[:limit]
    )


def _script_source_groups(scripts):
    labels = dict(PracticeScript.SOURCE_CHOICES)
    source_order = [
        PracticeScript.SOURCE_GENERATED,
        PracticeScript.SOURCE_UPLOADED,
        PracticeScript.SOURCE_USER,
        PracticeScript.SOURCE_BUILTIN,
        PracticeScript.SOURCE_IMPORTED,
    ]
    rows = list(scripts)
    groups = []
    for source in source_order:
        group_scripts = [script for script in rows if script.source == source]
        if group_scripts:
            groups.append(
                {
                    "source": source,
                    "label": labels.get(source, source.title()),
                    "scripts": group_scripts,
                }
            )
    other_scripts = [
        script for script in rows if script.source not in set(source_order)
    ]
    if other_scripts:
        groups.append({"source": "other", "label": "Other", "scripts": other_scripts})
    return groups


def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int]:
    raw = range_header.split("=", 1)[1].split(",", 1)[0].strip()
    start_raw, _, end_raw = raw.partition("-")
    if start_raw:
        start = int(start_raw)
        end = int(end_raw) if end_raw else file_size - 1
    else:
        suffix_length = int(end_raw or "0")
        start = max(0, file_size - suffix_length)
        end = file_size - 1
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    return start, end


def _submission_id(request) -> uuid.UUID | None:
    raw = str(request.headers.get("X-Idempotency-Key", "") or "").strip()
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError):
        return None


def _parse_date_param(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return fallback


def _format_metric(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.3g}{suffix}"


def _connect_autumn(request, settings_obj: PracticeSettings) -> None:
    base_url = normalize_base_url(
        request.POST.get("autumn_base_url")
        or settings_obj.autumn_base_url
        or django_settings.DEFAULT_AUTUMN_BASE_URL
    )
    username = (request.POST.get("autumn_username") or "").strip()
    password = request.POST.get("autumn_password") or ""
    if not username or not password:
        messages.error(request, "Enter your Autumn username and password.")
        return

    try:
        client = AutumnClient(base_url)
        token = client.authenticate(username, password)
    except AutumnError as exc:
        messages.error(request, f"Autumn login failed: {exc}")
        return

    settings_obj.autumn_base_url = client.base_url
    settings_obj.set_secret("autumn_token", token)
    refresh_message = _refresh_autumn_project_choices(client, settings_obj)
    settings_obj.save()
    messages.success(request, f"Autumn connected.{refresh_message}")


def _autumn_form_options(
    settings_obj: PracticeSettings,
    project: str | None = None,
) -> dict[str, list[str]]:
    token = settings_obj.get_secret("autumn_token") or ""
    selected_project = (project or settings_obj.autumn_project or "").strip()
    if not token:
        return {
            "autumn_projects": [selected_project] if selected_project else [],
            "autumn_subproject_options": list(settings_obj.autumn_subprojects or []),
        }

    try:
        client = _autumn_token_client(settings_obj)
        projects = client.list_projects()
    except AutumnError:
        projects = []

    if not selected_project and projects:
        selected_project = projects[0]
    if selected_project and selected_project not in projects:
        projects = [selected_project, *projects]

    subprojects = []
    if selected_project:
        try:
            subprojects = _autumn_token_client(settings_obj).list_subprojects(selected_project)
        except AutumnError:
            subprojects = []
    if not subprojects:
        subprojects = list(settings_obj.autumn_subprojects or [])

    return {
        "autumn_projects": projects,
        "autumn_subproject_options": subprojects,
    }


def _refresh_autumn_project_choices(client: AutumnClient, settings_obj: PracticeSettings) -> str:
    try:
        projects = client.list_projects()
    except AutumnError as exc:
        return f" Project refresh failed: {exc}"
    if not projects:
        return " No Autumn projects were returned."

    if not settings_obj.autumn_project or settings_obj.autumn_project not in projects:
        settings_obj.autumn_project = projects[0]

    try:
        available_subprojects = client.list_subprojects(settings_obj.autumn_project)
    except AutumnError:
        available_subprojects = []
    if available_subprojects and settings_obj.autumn_subprojects:
        available = set(available_subprojects)
        settings_obj.autumn_subprojects = [
            item for item in settings_obj.autumn_subprojects if item in available
        ]
    return f" Project: {settings_obj.autumn_project}."


def _autumn_token_client(settings_obj: PracticeSettings) -> AutumnClient:
    token = settings_obj.get_secret("autumn_token") or ""
    if not token:
        raise AutumnError("Store an Autumn token first.")
    return AutumnClient(settings_obj.autumn_base_url, token)


def _start_autumn_timer(
    request,
    settings_obj: PracticeSettings,
    note: str | None = None,
    notify: bool = True,
) -> dict[str, object]:
    try:
        payload = _autumn_client(settings_obj).start_timer(
            settings_obj.autumn_project,
            settings_obj.autumn_subprojects or [],
            note=note or "SpeechPractice practice",
        )
        session = payload.get("session") or {}
        session_id = session.get("id")
        if not session_id:
            raise AutumnError("Autumn did not return a timer id")
        settings_obj.autumn_active_session_id = int(session_id)
        settings_obj.save(update_fields=["autumn_active_session_id", "updated_at"])
        message = f"Autumn timer started for {settings_obj.autumn_project}."
        if notify:
            messages.success(request, message)
        return {
            "ok": True,
            "active": True,
            "message": message,
            "button_label": "Stop Autumn",
            "button_name": "stop_autumn_timer",
        }
    except (AutumnError, ValueError, TypeError) as exc:
        message = f"Autumn start failed: {exc}"
        if notify:
            messages.error(request, message)
        return {
            "ok": False,
            "active": bool(settings_obj.autumn_active_session_id),
            "message": message,
        }


def _stop_autumn_timer(
    request,
    settings_obj: PracticeSettings,
    note: str | None = None,
    notify: bool = True,
) -> dict[str, object]:
    try:
        payload = _autumn_client(settings_obj).stop_timer(
            session_id=settings_obj.autumn_active_session_id,
            project=settings_obj.autumn_project,
            note=note or _latest_autumn_note(),
        )
        settings_obj.autumn_active_session_id = None
        settings_obj.save(update_fields=["autumn_active_session_id", "updated_at"])
        duration = payload.get("duration")
        if duration is not None:
            message = f"Autumn timer stopped after {float(duration):.1f} minutes."
        else:
            message = "Autumn timer stopped."
        if notify:
            messages.success(request, message)
        return {
            "ok": True,
            "active": False,
            "message": message,
            "button_label": "Start Autumn",
            "button_name": "start_autumn_timer",
        }
    except (AutumnError, ValueError, TypeError) as exc:
        message = f"Autumn stop failed: {exc}"
        if notify:
            messages.error(request, message)
        return {
            "ok": False,
            "active": bool(settings_obj.autumn_active_session_id),
            "message": message,
        }


def _autumn_timer_context(
    settings_obj: PracticeSettings,
    selected_script: PracticeScript | None = None,
) -> dict[str, object]:
    has_token = settings_obj.has_secret("autumn_token")
    configured = has_token and bool(settings_obj.autumn_project)
    active = configured and bool(settings_obj.autumn_active_session_id)
    note = "SpeechPractice practice"
    if selected_script is not None:
        note = f"SpeechPractice: {selected_script.title}"
    return {
        "configured": configured,
        "connected": has_token,
        "active": active,
        "project": settings_obj.autumn_project,
        "subprojects": settings_obj.autumn_subprojects or [],
        "note": note,
    }


def _safe_next_url(request, raw_url: str | None) -> str:
    if raw_url and url_has_allowed_host_and_scheme(
        raw_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return raw_url
    return ""


def _autumn_client(settings_obj: PracticeSettings) -> AutumnClient:
    token = settings_obj.get_secret("autumn_token") or ""
    if not token:
        raise AutumnError("Store an Autumn token first.")
    if not settings_obj.autumn_project:
        raise AutumnError("Choose an Autumn project first.")
    return AutumnClient(settings_obj.autumn_base_url, token)


def _latest_autumn_note(user) -> str:
    try:
        session = (
            PracticeSession.objects.filter(user=user)
            .exclude(timestamp="")
            .order_by("-timestamp", "-id")
            .first()
        )
    except (OperationalError, ProgrammingError):
        session = None
    if session is None:
        return "SpeechPractice practice"
    parts = [f"SpeechPractice: {session.script_name}"]
    if session.score is not None:
        parts.append(f"Score {session.score:.2f}/5")
    if session.wer is not None:
        parts.append(f"WER {session.wer:.1%}")
    if session.cer is not None:
        parts.append(f"CER {session.cer:.1%}")
    if session.clarity is not None:
        parts.append(f"Clarity {session.clarity:.1%}")
    if session.artic_rate is not None:
        parts.append(f"Rate {session.artic_rate:.0f} wpm")
    return " | ".join(parts)


def _whisper_settings_snapshot(settings_obj: PracticeSettings) -> tuple:
    field_names = (
        "whisper_model_name",
        "whisper_device",
        "whisper_preset",
        "whisper_language",
        "whisper_timestamps",
        "whisper_beam_size",
        "whisper_temperature",
        "whisper_no_speech_threshold",
        "whisper_condition_on_previous_text",
        "whisper_chunk_seconds",
    )
    return tuple(
        PracticeSettings.objects.filter(pk=settings_obj.pk)
        .values_list(*field_names)
        .get()
    )


def _card_evidence_rows(card: ImprovementCard, limit: int = 8) -> list[dict]:
    try:
        rows = list(
            SessionError.objects.filter(user=card.user)
            .exclude(ref_token__isnull=True)
            .exclude(ref_token="")
            .filter(op__in=["sub", "del"])
            .order_by("-timestamp", "-id")[:800]
        )
    except (OperationalError, ProgrammingError):
        return []
    evidence = []
    sessions: dict[int, PracticeSession | None] = {}
    for error in rows:
        session_id = int(error.session_id)
        if session_id not in sessions:
            sessions[session_id] = PracticeSession.objects.filter(user=card.user, pk=session_id).first()
        session = sessions[session_id]
        if session is None or not _error_matches_card(card, error, session):
            continue
        snippet = _evidence_snippet(card, error, session)
        evidence.append(
            {
                "script_name": session.script_name or error.script_name or "Practice session",
                "timestamp": session.timestamp or error.timestamp,
                "error_kind": error.error_kind or error.op or "speech error",
                "expected": error.ref_token or "[nothing]",
                "heard": error.hyp_token or "[nothing]",
                "before": snippet["before"],
                "focus": snippet["focus"],
                "after": snippet["after"],
            }
        )
        if len(evidence) >= limit:
            break
    return evidence or _fallback_card_evidence_rows(card, limit=limit)


def _error_matches_card(
    card: ImprovementCard,
    error: SessionError,
    session: PracticeSession,
) -> bool:
    target = clean_text_for_alignment(card.target_key)
    ref_token = clean_text_for_alignment(error.ref_token or "")
    if not target or not ref_token:
        return False

    if card.kind == ImprovementCard.KIND_WORD:
        return ref_token == target

    if card.kind == ImprovementCard.KIND_PHRASE:
        script = clean_text_for_alignment(session.script_text or "")
        target_words = set(target.split())
        return target in script and ref_token in target_words

    if card.kind == ImprovementCard.KIND_SOUND:
        return card.target_key.strip().upper() in _word_to_phoneme_symbols(ref_token)

    if card.kind == ImprovementCard.KIND_POSITION:
        bucket = _position_bucket(
            error.ref_local_start,
            error.ref_local_end,
            error.ref_token_len,
        )
        return bucket == target

    if card.kind == ImprovementCard.KIND_CHARACTER:
        return (error.error_kind or "").strip().lower() == target

    return False


def _evidence_snippet(
    card: ImprovementCard,
    error: SessionError,
    session: PracticeSession,
) -> dict[str, str]:
    script = clean_text_for_alignment(session.script_text or "")
    target = clean_text_for_alignment(card.target_key)
    ref_token = clean_text_for_alignment(error.ref_token or "")
    focus = target if card.kind == ImprovementCard.KIND_PHRASE and target in script else ref_token
    return _snippet_parts(script, focus, error.ref_start, error.ref_end)


def _fallback_card_evidence_rows(card: ImprovementCard, limit: int = 8) -> list[dict]:
    if card.kind not in {ImprovementCard.KIND_WORD, ImprovementCard.KIND_PHRASE}:
        return []
    target = clean_text_for_alignment(card.target_key)
    if not target:
        return []
    try:
        sessions = list(
            PracticeSession.objects.filter(user=card.user)
            .exclude(score__isnull=True)
            .exclude(script_text="")
            .order_by("-timestamp", "-id")[:300]
        )
    except (OperationalError, ProgrammingError):
        return []

    evidence = []
    for session in sessions:
        script = clean_text_for_alignment(session.script_text or "")
        transcript = clean_text_for_alignment(session.transcript or "")
        if target not in script:
            continue
        if card.kind == ImprovementCard.KIND_WORD:
            transcript_has_target = target in set(transcript.split())
        else:
            transcript_has_target = target in transcript
        if transcript_has_target:
            continue
        snippet = _snippet_parts(script, target)
        evidence.append(
            {
                "script_name": session.script_name or "Practice session",
                "timestamp": session.timestamp,
                "error_kind": "target missing in transcript",
                "expected": card.target_key,
                "heard": "[target not found]",
                "before": snippet["before"],
                "focus": snippet["focus"],
                "after": snippet["after"],
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


def _snippet_parts(
    text: str,
    focus: str,
    fallback_start: int | None = None,
    fallback_end: int | None = None,
    radius: int = 68,
) -> dict[str, str]:
    clean_focus = clean_text_for_alignment(focus)
    start = text.find(clean_focus) if clean_focus else -1
    end = start + len(clean_focus) if start >= 0 else -1
    if start < 0 and fallback_start is not None:
        start = max(0, min(int(fallback_start), len(text)))
        fallback = int(fallback_end) if fallback_end is not None else start + len(clean_focus)
        end = max(start, min(fallback, len(text)))
    if start < 0 or end <= start:
        start, end = 0, min(len(text), radius)

    before_start = max(0, start - radius)
    after_end = min(len(text), end + radius)
    before = text[before_start:start].strip()
    after = text[end:after_end].strip()
    if before_start > 0:
        before = f"... {before}"
    if after_end < len(text):
        after = f"{after} ..."
    return {
        "before": before,
        "focus": text[start:end].strip() or clean_focus or "target",
        "after": after,
    }


def _mistake_lines(session: PracticeSession) -> list[str]:
    lines = []
    errors = SessionError.objects.filter(user=session.user, session_id=session.pk).order_by("id")[:200]
    for error in errors:
        label = error.error_kind or error.op or "error"
        expected = error.ref_token or "[nothing]"
        heard = error.hyp_token or "[nothing]"
        lines.append(f"{label}: expected {expected}; heard {heard}")
    return lines
