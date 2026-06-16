from __future__ import annotations

import re
import json
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path

from django.contrib import messages
from django.core.files.uploadedfile import UploadedFile
from django.db import models
from django.db.utils import OperationalError, ProgrammingError
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from autumn_client import AutumnClient, AutumnError

from .forms import AccountSettingsForm, BulkScriptImportForm, PracticeRunForm, PracticeScriptForm, TranscriptEditForm
from .models import GeneratedPracticeScript, ImprovementCard, PracticeReview, PracticeScript, PracticeSession, PracticeSettings, ScoringJob, SessionError
from .services.analytics import (
    active_scoring_jobs,
    dashboard_stats,
    recent_scoring_jobs,
    recent_sessions,
    refresh_improvement_cards,
    score_distribution,
    progress_series,
    script_name_options,
    today_queue,
    trend_summary_for_range,
    trend_summary,
)
from .services.jobs import create_free_speak_job, create_scoring_job, enqueue_scoring_job, job_status_context
from .services.scoring import _replace_session_errors, recording_upload_dir, score_transcript
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
from .services.script_generation import generate_script_draft, script_generation_provider_choices
from .services.transcription import TranscriptResult
from .services.transcription import clear_local_whisper_cache, provider_label


def dashboard(request):
    summary = trend_summary()
    context = {
        "stats": dashboard_stats(),
        "recent_sessions": recent_sessions(),
        "summary": summary,
        "today_queue": today_queue(),
        "active_jobs": active_scoring_jobs(),
        "recent_jobs": recent_scoring_jobs(),
        "score_distribution": score_distribution(),
    }
    return render(request, "practice/dashboard.html", context)


def progress(request):
    end_dt = _parse_date_param(request.GET.get("end"), datetime.now())
    start_dt = _parse_date_param(request.GET.get("start"), end_dt - timedelta(days=30))
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    script_name = (request.GET.get("script") or "").strip() or None
    points = progress_series(start_dt=start_dt, end_dt=end_dt, script_name=script_name)
    summary = trend_summary_for_range(
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
            "script_options": script_name_options(),
            "points": points,
            "points_json": json.dumps(points),
            "summary": summary,
        },
    )


def practice_run(request):
    quick_queue = today_queue(limit=4)
    builtin_drills = _builtin_drills(limit=5)
    requested_mode = request.POST.get("mode") if request.method == "POST" else request.GET.get("mode")
    if not requested_mode and request.method == "GET" and request.GET.get("card"):
        requested_mode = PracticeRunForm.MODE_QUICK
    mode = requested_mode or PracticeRunForm.MODE_SCRIPT
    script_kind = _script_kind_for_mode(mode)
    if request.method == "POST":
        form = PracticeRunForm(request.POST, request.FILES, script_kind=script_kind)
        if form.is_valid():
            mode = form.cleaned_data.get("mode") or PracticeRunForm.MODE_SCRIPT
            script = form.cleaned_data.get("script")
            card = form.cleaned_data.get("card")
            audio = form.cleaned_data["audio"]
            provider = form.cleaned_data.get("provider") or None
            if audio is None:
                form.add_error("audio", "Record or upload audio before scoring.")
            else:
                audio_path = _save_uploaded_audio(audio, script.title if script else "free-speak")
                if mode == PracticeRunForm.MODE_FREE:
                    job = create_free_speak_job(
                        audio_path=str(audio_path),
                        provider=provider or "local_whisper",
                    )
                else:
                    job = create_scoring_job(
                        script=script,
                        audio_path=str(audio_path),
                        provider=provider or "local_whisper",
                        card=card,
                    )
                enqueue_scoring_job(job)
                focus_label = f" for {card.title}" if card else ""
                mode_label = "Free Speak transcription" if mode == PracticeRunForm.MODE_FREE else "Scoring"
                messages.success(
                    request,
                    f"{mode_label} queued{focus_label} with {provider_label(provider)}.",
                )
                return redirect("practice:scoring_job", pk=job.pk)
    else:
        script_id = request.GET.get("script")
        card_id = request.GET.get("card")
        initial_card = None
        initial_script = None
        if card_id:
            initial_card = (
                ImprovementCard.objects.exclude(status=ImprovementCard.STATUS_PAUSED)
                .filter(pk=card_id)
                .first()
            )
        if mode == PracticeRunForm.MODE_QUICK:
            if script_id:
                initial_script = PracticeScript.objects.filter(
                    pk=script_id,
                    active=True,
                    practice_kind=PracticeScript.KIND_DRILL,
                ).first()
            if initial_script is None and initial_card is not None:
                initial_script = _latest_script_for_card(initial_card)
            if initial_script is None and quick_queue:
                initial_card = quick_queue[0].card
                initial_script = quick_queue[0].script
            if initial_script is None:
                initial_script = builtin_drills[0] if builtin_drills else None
        elif script_id:
            initial_script = PracticeScript.objects.filter(
                pk=script_id,
                active=True,
                practice_kind=PracticeScript.KIND_READING,
            ).first()
        elif initial_card is not None:
            initial_script = _latest_script_for_card(initial_card)
        form = PracticeRunForm(
            initial_script=initial_script,
            initial_card=initial_card,
            script_kind=script_kind,
            initial={"mode": mode},
        )

    selected_script = None
    script_value = form["script"].value()
    if script_value:
        selected_query = PracticeScript.objects.filter(pk=script_value)
        if script_kind:
            selected_query = selected_query.filter(practice_kind=script_kind)
        selected_script = selected_query.first()
    if selected_script is None:
        fallback_scripts = PracticeScript.objects.filter(active=True)
        if script_kind:
            fallback_scripts = fallback_scripts.filter(practice_kind=script_kind)
        selected_script = fallback_scripts.first()
    focus_card = None
    card_value = form["card"].value()
    if card_value:
        focus_card = ImprovementCard.objects.filter(pk=card_value).first()
    context = {
        "form": form,
        "selected_script": selected_script,
        "focus_card": focus_card,
        "quick_queue": quick_queue,
        "builtin_drills": builtin_drills,
        "generation_provider_choices": script_generation_provider_choices(),
    }
    return render(request, "practice/practice_run.html", context)


def session_list(request):
    sessions = PracticeSession.objects.all()[:80]
    return render(request, "practice/session_list.html", {"sessions": sessions})


def session_detail(request, pk: int):
    session = get_object_or_404(PracticeSession, pk=pk)
    if request.method == "POST":
        if request.POST.get("action") == "clear_transcript":
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
            SessionError.objects.filter(session_id=session.pk).delete()
            messages.success(request, "Transcript cleared.")
            return redirect("practice:session_detail", pk=session.pk)

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

    highlighted = highlighted_session_text(session)
    mistake_lines = _mistake_lines(session)
    return render(
        request,
        "practice/session_detail.html",
        {
            "session": session,
            "form": form,
            "highlighted": highlighted,
            "has_audio": audio_exists(session),
            "mistake_lines": mistake_lines,
        },
    )


def session_report(request, pk: int):
    session = get_object_or_404(PracticeSession, pk=pk)
    errors = SessionError.objects.filter(session_id=session.pk).order_by("id")[:200]
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


@require_POST
def session_delete(request, pk: int):
    session = get_object_or_404(PracticeSession, pk=pk)
    script_name = session.script_name
    audio_deleted = _delete_audio_file(session.audio_path)
    SessionError.objects.filter(session_id=session.pk).delete()
    PracticeReview.objects.filter(legacy_session_id=session.pk).delete()
    ScoringJob.objects.filter(legacy_session_id=session.pk).update(legacy_session_id=None)
    session.delete()
    suffix = " and deleted the audio file" if audio_deleted else ""
    messages.success(request, f"Deleted recording entry for {script_name}{suffix}.")
    return redirect("practice:sessions")


def session_audio(request, pk: int):
    session = get_object_or_404(PracticeSession, pk=pk)
    path = Path(session.audio_path or "")
    if not path.exists() or not path.is_file():
        raise Http404("Audio file not found.")
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    file_size = path.stat().st_size
    range_header = request.headers.get("Range", "")
    if range_header.startswith("bytes="):
        start, end = _parse_byte_range(range_header, file_size)
        length = end - start + 1
        with path.open("rb") as audio_file:
            audio_file.seek(start)
            payload = audio_file.read(length)
        response = HttpResponse(payload, status=206, content_type=content_type)
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response["Content-Length"] = str(length)
    else:
        response = FileResponse(path.open("rb"), as_attachment=False, content_type=content_type)
        response["Content-Length"] = str(file_size)
    response["Accept-Ranges"] = "bytes"
    return response


def scoring_job_detail(request, pk: int):
    job = get_object_or_404(ScoringJob, pk=pk)
    session = None
    if job.legacy_session_id:
        session = PracticeSession.objects.filter(pk=job.legacy_session_id).first()
    context = {
        "job": job,
        "session": session,
        **job_status_context(job),
    }
    return render(request, "practice/scoring_job.html", context)


def scoring_job_status(request, pk: int):
    job = get_object_or_404(ScoringJob, pk=pk)
    session_url = ""
    if job.legacy_session_id:
        session_url = reverse("practice:session_detail", args=[job.legacy_session_id])
    return JsonResponse(
        {
            "id": job.pk,
            "status": job.status,
            "status_label": job.get_status_display(),
            "is_pending": job.status in {ScoringJob.STATUS_QUEUED, ScoringJob.STATUS_RUNNING},
            "is_done": job.status == ScoringJob.STATUS_SUCCEEDED,
            "is_failed": job.status == ScoringJob.STATUS_FAILED,
            "partial_transcript": job.partial_transcript or "",
            "error_message": job.error_message or "",
            "session_url": session_url,
        }
    )


@require_POST
def retry_scoring_job(request, pk: int):
    job = get_object_or_404(ScoringJob, pk=pk)
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
    scripts = PracticeScript.objects.filter(practice_kind=kind_filter)
    source_filter = (request.GET.get("source") or "").strip()
    total_count = scripts.count()
    if source_filter:
        scripts = scripts.filter(source=source_filter)
    groups = _script_source_groups(scripts)
    kind_counts = {
        row["practice_kind"]: row["count"]
        for row in PracticeScript.objects.values("practice_kind").annotate(count=models.Count("id"))
    }
    source_counts = {
        row["source"]: row["count"]
        for row in PracticeScript.objects.filter(practice_kind=kind_filter)
        .values("source")
        .annotate(count=models.Count("id"))
    }
    kind_tabs = [
        {
            "value": value,
            "label": "Reading scripts" if value == PracticeScript.KIND_READING else "Drills",
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
        },
    )


def script_create(request):
    if request.method == "POST":
        form = PracticeScriptForm(request.POST)
        if form.is_valid():
            form.save()
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
    script = get_object_or_404(PracticeScript, pk=pk)
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


def script_delete(request, pk: int):
    script = get_object_or_404(PracticeScript, pk=pk)
    if request.method == "POST":
        title = script.title
        script.delete()
        messages.success(request, f"Deleted script: {title}.")
        return redirect("practice:scripts")
    return render(request, "practice/script_confirm_delete.html", {"script": script})


def script_preview(request, pk: int):
    script = get_object_or_404(PracticeScript, pk=pk, active=True)
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
    if request.GET.get("refresh") == "1":
        created = refresh_improvement_cards()
        messages.success(request, f"Cards refreshed. {created} new focus areas created.")
        return redirect("practice:cards")
    cards = ImprovementCard.objects.all()
    return render(request, "practice/card_list.html", {"cards": cards})


def card_detail(request, pk: int):
    card = get_object_or_404(ImprovementCard, pk=pk)
    reviews = card.reviews.all()[:8]
    generated_scripts = PracticeScript.objects.filter(
        generation_records__card=card,
        practice_kind=PracticeScript.KIND_DRILL,
    ).distinct()[:8]
    return render(
        request,
        "practice/card_detail.html",
        {
            "card": card,
            "reviews": reviews,
            "generated_scripts": generated_scripts,
            "generation_provider_choices": script_generation_provider_choices(),
        },
    )


@require_POST
def generate_script_for_card(request, pk: int):
    card = get_object_or_404(ImprovementCard, pk=pk)
    provider = request.POST.get("provider") or None
    try:
        draft = generate_script_draft(card, provider_name=provider)
    except Exception as exc:
        messages.error(request, f"Script generation failed: {exc}")
        return redirect("practice:card_detail", pk=card.pk)
    script = PracticeScript.objects.create(
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
        card=card,
        script=script,
        model_provider=draft.provider,
        prompt_snapshot=draft.prompt_snapshot,
    )
    messages.success(request, "Generated a starter practice script for this card.")
    if request.POST.get("next") == "practice":
        return redirect(f"{reverse('practice:practice')}?mode=quick&script={script.pk}&card={card.pk}")
    return redirect("practice:scripts")


def account_settings(request):
    settings_obj = PracticeSettings.load()
    if request.method == "POST":
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
            try:
                payload = _autumn_client(settings_obj).start_timer(
                    settings_obj.autumn_project,
                    settings_obj.autumn_subprojects or [],
                    note=request.POST.get("autumn_note") or "SpeechPractice practice",
                )
                session = payload.get("session") or {}
                session_id = session.get("id")
                if not session_id:
                    raise AutumnError("Autumn did not return a timer id")
                settings_obj.autumn_active_session_id = int(session_id)
                settings_obj.save(update_fields=["autumn_active_session_id", "updated_at"])
                messages.success(request, f"Autumn timer started for {settings_obj.autumn_project}.")
            except (AutumnError, ValueError, TypeError) as exc:
                messages.error(request, f"Autumn start failed: {exc}")
            return redirect("practice:account")

        if "stop_autumn_timer" in request.POST:
            try:
                payload = _autumn_client(settings_obj).stop_timer(
                    session_id=settings_obj.autumn_active_session_id,
                    project=settings_obj.autumn_project,
                    note=request.POST.get("autumn_note") or _latest_autumn_note(),
                )
                settings_obj.autumn_active_session_id = None
                settings_obj.save(update_fields=["autumn_active_session_id", "updated_at"])
                duration = payload.get("duration")
                if duration is not None:
                    messages.success(request, f"Autumn timer stopped after {float(duration):.1f} minutes.")
                else:
                    messages.success(request, "Autumn timer stopped.")
            except (AutumnError, ValueError, TypeError) as exc:
                messages.error(request, f"Autumn stop failed: {exc}")
            return redirect("practice:account")

        old_whisper = _whisper_settings_snapshot(settings_obj)
        form = AccountSettingsForm(request.POST, instance=settings_obj)
        if form.is_valid():
            form.save()
            if old_whisper != _whisper_settings_snapshot(settings_obj):
                clear_local_whisper_cache()
            messages.success(request, "Account settings saved.")
            return redirect("practice:account")
    else:
        form = AccountSettingsForm(instance=settings_obj)

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
        "autumn_note": _latest_autumn_note(),
    }
    return render(request, "practice/account.html", context)


def _latest_script_for_card(card: ImprovementCard) -> PracticeScript | None:
    generated = (
        GeneratedPracticeScript.objects.select_related("script")
        .filter(card=card, script__active=True, script__practice_kind=PracticeScript.KIND_DRILL)
        .order_by("-created_at")
        .first()
    )
    return generated.script if generated and generated.script else None


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


def _save_uploaded_audio(audio: UploadedFile, script_title: str):
    ext = _safe_extension(audio.name)
    stamp = timezone.localtime().strftime("%Y%m%d_%H%M%S")
    title_slug = re.sub(r"[^a-zA-Z0-9]+", "-", script_title).strip("-").lower()[:40]
    filename = f"{stamp}_{title_slug or 'practice'}{ext}"
    destination = recording_upload_dir() / filename
    with destination.open("wb") as out:
        for chunk in audio.chunks():
            out.write(chunk)
    return destination


def _safe_extension(filename: str) -> str:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".webm"
    return suffix if suffix in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".txt"} else ".webm"


def _delete_audio_file(audio_path: str | None) -> bool:
    if not audio_path:
        return False
    path = Path(audio_path)
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


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


def _autumn_client(settings_obj: PracticeSettings) -> AutumnClient:
    token = settings_obj.get_secret("autumn_token") or ""
    if not token:
        raise AutumnError("Store an Autumn token first.")
    if not settings_obj.autumn_project:
        raise AutumnError("Choose an Autumn project first.")
    return AutumnClient(settings_obj.autumn_base_url, token)


def _latest_autumn_note() -> str:
    try:
        session = PracticeSession.objects.exclude(timestamp="").order_by("-timestamp", "-id").first()
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


def _mistake_lines(session: PracticeSession) -> list[str]:
    lines = []
    errors = SessionError.objects.filter(session_id=session.pk).order_by("id")[:200]
    for error in errors:
        label = error.error_kind or error.op or "error"
        expected = error.ref_token or "[nothing]"
        heard = error.hyp_token or "[nothing]"
        lines.append(f"{label}: expected {expected}; heard {heard}")
    return lines
