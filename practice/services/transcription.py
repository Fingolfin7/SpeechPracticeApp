from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from practice.models import PracticeSettings


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    provider: str
    segments: list[dict[str, Any]]
    raw: dict[str, Any]


class TranscriptionProvider(Protocol):
    name: str

    def transcribe(self, audio_path: str) -> TranscriptResult:
        ...


class LocalWhisperProvider:
    name = "local_whisper"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        language: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.WHISPER_MODEL_NAME
        self.device = device or settings.WHISPER_DEVICE
        self.language = language or settings.WHISPER_LANGUAGE

    def transcribe(self, audio_path: str) -> TranscriptResult:
        model = _load_whisper_model(self.model_name, self.device)
        options: dict[str, Any] = {
            "beam_size": settings.WHISPER_BEAM_SIZE,
            "temperature": settings.WHISPER_TEMPERATURE,
            "condition_on_previous_text": settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            "no_speech_threshold": settings.WHISPER_NO_SPEECH_THRESHOLD,
        }
        if self.language and self.language != "auto":
            options["language"] = self.language
        result = model.transcribe(audio_path, **options)
        return TranscriptResult(
            text=str(result.get("text", "")).strip(),
            provider=self.name,
            segments=list(result.get("segments") or []),
            raw=dict(result),
        )


class OpenAITranscriptionProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        app_settings = _practice_settings()
        self.model = model or (app_settings.openai_transcription_model if app_settings else None) or settings.OPENAI_TRANSCRIPTION_MODEL
        self.api_key = (app_settings.get_secret("openai_api_key") if app_settings else None) or settings.OPENAI_API_KEY

    def transcribe(self, audio_path: str) -> TranscriptResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI transcription.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI transcription.") from exc

        client = OpenAI(api_key=self.api_key)
        with Path(audio_path).open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        text = getattr(response, "text", "") or ""
        return TranscriptResult(
            text=str(text).strip(),
            provider=self.name,
            segments=[],
            raw={"model": self.model},
        )


class UploadedTranscriptProvider:
    """
    Test/development provider that treats a sibling .txt file as the transcript.
    Useful for smoke-testing the scoring flow without loading Whisper.
    """

    name = "uploaded_transcript"

    def transcribe(self, audio_path: str) -> TranscriptResult:
        path = Path(audio_path)
        sidecar = path if path.suffix.lower() == ".txt" else path.with_suffix(".txt")
        if not sidecar.exists():
            raise RuntimeError(f"Missing transcript sidecar: {sidecar}")
        return TranscriptResult(
            text=sidecar.read_text(encoding="utf-8").strip(),
            provider=self.name,
            segments=[],
            raw={"sidecar": str(sidecar)},
        )


@lru_cache(maxsize=4)
def _load_whisper_model(model_name: str, device: str):
    import whisper

    resolved_device = _resolve_device(device)
    return whisper.load_model(model_name, device=resolved_device)


def _resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        return "cuda" if _has_cuda() else "cpu"
    if device == "auto":
        return "cuda" if _has_cuda() else "cpu"
    return device


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_transcription_provider(provider_name: str | None = None) -> TranscriptionProvider:
    app_settings = _practice_settings()
    provider = provider_name or (app_settings.transcription_provider if app_settings else None) or settings.TRANSCRIPTION_PROVIDER
    if provider == "local_whisper":
        return LocalWhisperProvider()
    if provider == "openai":
        return OpenAITranscriptionProvider()
    if provider == "uploaded_transcript":
        return UploadedTranscriptProvider()
    raise ValueError(f"Unknown transcription provider: {provider}")


def provider_label(provider_name: str | None = None) -> str:
    app_settings = _practice_settings()
    provider = provider_name or (app_settings.transcription_provider if app_settings else None) or settings.TRANSCRIPTION_PROVIDER
    return {
        "local_whisper": "Local Whisper",
        "openai": "OpenAI transcription",
        "uploaded_transcript": "Transcript sidecar",
    }.get(provider, provider)


def _practice_settings() -> PracticeSettings | None:
    try:
        return PracticeSettings.load()
    except (OperationalError, ProgrammingError):
        return None
