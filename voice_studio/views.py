import threading
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from . import service

_active_request = threading.Lock()


@ensure_csrf_cookie
def index(request):
    return render(request, "voice_studio/index.html")


@require_POST
def evaluate(request):
    upload = request.FILES.get("audio")
    key = request.POST.get("api_key", "").strip()
    if not key or len(key) > 512:
        return JsonResponse({"status": "error", "reason": "Enter your OpenAI API key before sending."}, status=400)
    if not upload or not upload.size or upload.size > service.MAX_UPLOAD:
        return JsonResponse({"status": "error", "reason": "Record up to two minutes of audio (maximum 10 MB)."}, status=400)
    if not _active_request.acquire(blocking=False):
        return JsonResponse({"status": "error", "reason": "Feedback is already in progress. Please wait."}, status=409)
    try:
        wav = service.to_wav(upload)
        result = service.ask_model(key, wav)
        return JsonResponse(result)
    except service.AudioError as exc:
        return JsonResponse({"status": "error", "reason": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"status": "error", "reason": "Feedback could not be completed. Check your key and connection, then try again."}, status=502)
    finally:
        _active_request.release()
