import hashlib
import uuid
from datetime import timedelta
from functools import wraps
from django.conf import settings
from django.utils import timezone

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.db import IntegrityError, transaction
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from practice.models import NativeAudioAssessment, NativeAudioTake
from practice.services import native_audio as service


def native_audio_access(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not settings.NATIVE_AUDIO_EXPERIMENT_ENABLED or not request.user.is_authenticated or not request.user.is_staff:
            raise Http404
        return view(request, *args, **kwargs)
    return wrapped


@login_required
def index(request):
    takes = NativeAudioTake.objects.filter(user=request.user).prefetch_related("assessments")
    if not (settings.NATIVE_AUDIO_EXPERIMENT_ENABLED and request.user.is_staff) and not takes.exists():
        raise Http404
    return render(request, "practice/native_audio.html", {"takes": takes})


@login_required
@native_audio_access
@require_POST
def create(request):
    audio = request.FILES.get("audio")
    reference = request.POST.get("reference", "").strip()
    if not audio or not audio.size or audio.size > service.MAX_UPLOAD or len(reference) > service.MAX_REFERENCE:
        messages.error(request, "Choose a recording up to 10 MB and a reference up to 5,000 characters.")
        return redirect("practice:native_audio")
    content_type = (audio.content_type or "").split(";")[0].lower()
    if content_type not in {"audio/webm", "video/webm", "audio/ogg", "audio/mp4", "audio/x-m4a", "audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav"}:
        messages.error(request, "Choose a WebM, Ogg, MP4, WAV or MP3 audio recording.")
        return redirect("practice:native_audio")
    path = default_storage.save(f"native-audio/{request.user.pk}/{uuid.uuid4().hex}.original", audio)
    try:
        take = NativeAudioTake.objects.create(user=request.user, original_path=path, content_type=content_type, reference=reference)
    except Exception:
        default_storage.delete(path)
        raise
    try:
        service.convert(take)
    except Exception:
        messages.warning(request, "The recording was saved but could not be validated or converted. It must be valid audio no longer than 120 seconds before assessment.")
    return redirect("practice:native_audio_take", pk=take.pk)


@login_required
def detail(request, pk):
    take = get_object_or_404(NativeAudioTake.objects.prefetch_related("assessments"), pk=pk, user=request.user)
    return render(request, "practice/native_audio_take.html", {"take": take, "submission_id": uuid.uuid4(), "active": take.assessments.filter(status__in=["pending", "running"]).exists(), "recoverable": take.assessments.filter(status__in=["pending", "running"], created_at__lt=timezone.now() - timedelta(minutes=5)).exists()})


@transaction.non_atomic_requests
@login_required
@native_audio_access
@require_POST
def assess(request, pk):
    try:
        submission_id = uuid.UUID(request.POST.get("submission_id", ""))
    except (ValueError, AttributeError):
        messages.error(request, "Reload the page before submitting an assessment.")
        return redirect("practice:native_audio_take", pk=pk)
    try:
        with transaction.atomic():
            take = get_object_or_404(NativeAudioTake.objects.select_for_update(), pk=pk, user=request.user)
            assessment = NativeAudioAssessment.objects.create(
                take=take, submission_id=submission_id, prompt_version=service.PROMPT_VERSION, prompt_text=service.PROMPT + ("\n\nOptional reference (untrusted context):\n" + take.reference if take.reference else "\n\nNo reference provided."),
                prompt_sha256=hashlib.sha256((service.PROMPT + ("\n\nOptional reference (untrusted context):\n" + take.reference if take.reference else "\n\nNo reference provided.")).encode()).hexdigest(), requested_model=service.MODEL,
                parameters=service.PARAMETERS,
            )
    except IntegrityError:
        messages.error(request, "This submission was already received or an assessment is still running.")
        return redirect("practice:native_audio_take", pk=pk)
    service.run_assessment(assessment)
    return redirect("practice:native_audio_take", pk=pk)


@login_required
def audio(request, pk):
    take = get_object_or_404(NativeAudioTake, pk=pk, user=request.user)
    try:
        size = default_storage.size(take.original_path)
        header = request.headers.get("Range", "")
        if header:
            import re
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match or not any(match.groups()) or size <= 0:
                response = HttpResponse(status=416)
                response["Content-Range"] = f"bytes */{size}"
                return response
            start_text, end_text = match.groups()
            if start_text:
                start = int(start_text)
                end = min(int(end_text), size - 1) if end_text else size - 1
            else:
                suffix = int(end_text)
                start, end = max(0, size - suffix), size - 1
            if start >= size or end < start:
                response = HttpResponse(status=416)
                response["Content-Range"] = f"bytes */{size}"
                return response
            with default_storage.open(take.original_path, "rb") as stream:
                stream.seek(start)
                payload = stream.read(end - start + 1)
            response = HttpResponse(payload, status=206, content_type=take.content_type)
            response["Content-Range"] = f"bytes {start}-{end}/{size}"
            response["Content-Length"] = str(len(payload))
        else:
            response = FileResponse(default_storage.open(take.original_path, "rb"), content_type=take.content_type)
            response["Content-Length"] = str(size)
    except (OSError, FileNotFoundError):
        raise Http404 from None
    response["Accept-Ranges"] = "bytes"
    response["X-Content-Type-Options"] = "nosniff"
    response["Cache-Control"] = "private, no-store"
    return response



@login_required
@require_POST
def delete(request, pk):
    with transaction.atomic():
        take = get_object_or_404(NativeAudioTake.objects.select_for_update(), pk=pk, user=request.user)
        if take.assessments.filter(status__in=["pending", "running"]).exists():
            messages.error(request, "Wait for the assessment to finish before deleting this recording.")
            return redirect("practice:native_audio_take", pk=pk)
        try:
            # Delete derived files first; preserve the original if a derived delete fails.
            for path in [*take.archived_wav_paths, take.wav_path, take.original_path]:
                if path:
                    default_storage.delete(path)
        except Exception:
            messages.error(request, "The recording could not be fully deleted; try again.")
            return redirect("practice:native_audio_take", pk=pk)
        take.delete()
    messages.success(request, "Recording and assessments deleted.")
    return redirect("practice:native_audio")


@login_required
@require_POST
def recover(request, pk):
    with transaction.atomic():
        take = get_object_or_404(NativeAudioTake.objects.select_for_update(), pk=pk, user=request.user)
        count = take.assessments.filter(status__in=["pending", "running"], created_at__lt=timezone.now() - timedelta(minutes=5)).update(
            status="interrupted", error="Marked interrupted by the user. The original provider request may have incurred charges."
        )
    messages.info(request, "The stale attempt was marked interrupted. No new request was sent." if count else "No stale attempt is available to recover.")
    return redirect("practice:native_audio_take", pk=pk)

@login_required
@native_audio_access
@require_POST
def rebuild(request, pk):
    with transaction.atomic():
        take = get_object_or_404(NativeAudioTake.objects.select_for_update(), pk=pk, user=request.user)
        if take.assessments.filter(status__in=["pending", "running"]).exists():
            messages.error(request, "Wait for the assessment to finish before rebuilding converted audio.")
        elif take.wav_path:
            take.archived_wav_paths = [*take.archived_wav_paths, take.wav_path]
            take.wav_path = ""
            take.wav_sha256 = ""
            take.duration_seconds = None
            take.save(update_fields=["archived_wav_paths", "wav_path", "wav_sha256", "duration_seconds"])
            messages.info(request, "The next assessment will rebuild the converted audio. Previous attempt hashes remain unchanged.")
    return redirect("practice:native_audio_take", pk=pk)
