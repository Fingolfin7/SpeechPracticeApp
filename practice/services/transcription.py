from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from practice.models import PracticeSettings
from practice.services.codex_auth import codex_access_token
from practice.services.local_whisper import transcribe_source

OPENAI_TIMESTAMPED_TRANSCRIPTION_MODEL = "whisper-1"


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    provider: str
    segments: list[dict[str, Any]]
    raw: dict[str, Any]


class TranscriptionProvider(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: str,
        partial_callback: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        ...


class LocalWhisperProvider:
    name = "local_whisper"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        language: str | None = None,
    ) -> None:
        app_settings = _practice_settings()
        self.app_settings = app_settings
        self.model_name = (
            model_name
            or (app_settings.whisper_model_name if app_settings else None)
            or settings.WHISPER_MODEL_NAME
        )
        self.device = (
            device
            or (app_settings.whisper_device if app_settings else None)
            or settings.WHISPER_DEVICE
        )
        self.language = (
            language
            or (app_settings.whisper_language if app_settings else None)
            or settings.WHISPER_LANGUAGE
        )

    def transcribe(
        self,
        audio_path: str,
        partial_callback: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        model = _load_whisper_model(self.model_name, self.device)
        options = _whisper_options(self.app_settings)
        if self.language and self.language != "auto":
            options["language"] = self.language
        result = transcribe_source(model, audio_path, options, partial_callback=partial_callback)
        text = str(result.get("text", "")).strip()
        if partial_callback and text:
            partial_callback(text)
        return TranscriptResult(
            text=text,
            provider=self.name,
            segments=list(result.get("segments") or []),
            raw=dict(result),
        )


class OpenAITranscriptionProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        app_settings = _practice_settings()
        self.app_settings = app_settings
        configured_model = (
            model
            or (app_settings.openai_transcription_model if app_settings else None)
            or settings.OPENAI_TRANSCRIPTION_MODEL
        )
        self.configured_model = configured_model
        self.model = OPENAI_TIMESTAMPED_TRANSCRIPTION_MODEL
        self.chunk_seconds = _chunk_seconds(app_settings)
        self.condition_on_previous_text = (
            bool(app_settings.whisper_condition_on_previous_text)
            if app_settings is not None
            else settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT
        )
        self.api_key = (app_settings.get_secret("openai_api_key") if app_settings else None) or settings.OPENAI_API_KEY

    def transcribe(
        self,
        audio_path: str,
        partial_callback: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI transcription.") from exc

        codex_token = codex_access_token(self.app_settings)
        if codex_token:
            try:
                return self._transcribe_with_client(
                    OpenAI,
                    audio_path,
                    api_key=codex_token,
                    auth_source="codex",
                    partial_callback=partial_callback,
                )
            except Exception as exc:
                if not self.api_key:
                    if _looks_like_missing_api_scope(exc):
                        raise RuntimeError(
                            "Codex auth was accepted by the OpenAI API, but this "
                            "token does not include the API scopes required for "
                            "OpenAI transcription. Add an OpenAI API key for "
                            "OpenAI transcription, or switch to Local Whisper."
                        ) from exc
                    if _looks_like_browser_challenge(exc):
                        raise RuntimeError(
                            "Codex auth reached ChatGPT, but the audio transcription "
                            "endpoint returned a browser challenge. Add an OpenAI API "
                            "key for OpenAI transcription, or switch to Local Whisper."
                        ) from exc
                    raise
        if not self.api_key:
            raise RuntimeError("OpenAI transcription requires a Codex login or an OpenAI API key.")
        return self._transcribe_with_client(
            OpenAI,
            audio_path,
            api_key=self.api_key,
            auth_source="api_key" if not codex_token else "api_key_fallback",
            partial_callback=partial_callback,
        )

    def _transcribe_with_client(
        self,
        openai_class,
        audio_path: str,
        *,
        api_key: str,
        auth_source: str,
        partial_callback: Callable[[str], None] | None = None,
        base_url: str | None = None,
    ) -> TranscriptResult:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai_class(**kwargs)
        result = self._transcribe_chunked(client, audio_path, partial_callback=partial_callback)
        text = result["text"]
        segments = result["segments"]
        raw = result["raw"]
        if partial_callback and text and not result.get("partial_sent"):
            partial_callback(str(text).strip())
        return TranscriptResult(
            text=str(text).strip(),
            provider=self.name,
            segments=segments,
            raw={
                "model": self.model,
                "configured_model": self.configured_model,
                "auth_source": auth_source,
                "response": raw,
            },
        )

    def _transcribe_chunked(
        self,
        client,
        audio_path: str,
        partial_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise RuntimeError("Install pydub to use chunked OpenAI transcription.") from exc

        audio = AudioSegment.from_file(audio_path)
        chunk_ms = max(1, int(self.chunk_seconds * 1000))
        if len(audio) <= chunk_ms:
            raw = self._transcribe_file(client, Path(audio_path), prompt="")
            return {
                "text": str(raw.get("text", "") or "").strip(),
                "segments": _openai_segments(raw),
                "raw": raw,
                "partial_sent": False,
            }

        text_parts: list[str] = []
        all_segments: list[dict[str, Any]] = []
        raw_chunks: list[dict[str, Any]] = []
        partial_sent = False
        with tempfile.TemporaryDirectory(prefix="speechpractice-openai-chunks-") as temp_dir:
            for index, start_ms in enumerate(range(0, len(audio), chunk_ms)):
                chunk = audio[start_ms : start_ms + chunk_ms]
                if len(chunk) <= 0:
                    continue
                chunk_path = Path(temp_dir) / f"chunk-{index:04d}.wav"
                chunk.export(chunk_path, format="wav")
                raw = self._transcribe_file(
                    client,
                    chunk_path,
                    prompt=self._chunk_prompt(text_parts),
                )
                raw_chunks.append(raw)
                chunk_text = str(raw.get("text", "") or "").strip()
                if chunk_text:
                    text_parts.append(chunk_text)
                _append_openai_segments(
                    all_segments,
                    _openai_segments(raw),
                    offset_seconds=start_ms / 1000.0,
                    chunk_duration_seconds=len(chunk) / 1000.0,
                )
                if partial_callback is not None:
                    try:
                        partial_callback(_join_transcript_parts(text_parts))
                        partial_sent = True
                    except Exception:
                        pass
        return {
            "text": _join_transcript_parts(text_parts),
            "segments": all_segments,
            "raw": {"chunks": raw_chunks},
            "partial_sent": partial_sent,
        }

    def _transcribe_file(self, client, path: Path, prompt: str = "") -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if prompt:
            request["prompt"] = prompt
        with path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(file=audio_file, **request)
        return _plain_data(response)

    def _chunk_prompt(self, text_parts: list[str]) -> str:
        if not self.condition_on_previous_text:
            return ""
        previous_text = _join_transcript_parts(text_parts)
        return previous_text[-500:].strip()


class UploadedTranscriptProvider:
    """
    Test/development provider that treats a sibling .txt file as the transcript.
    Useful for smoke-testing the scoring flow without loading Whisper.
    """

    name = "uploaded_transcript"

    def transcribe(
        self,
        audio_path: str,
        partial_callback: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        path = Path(audio_path)
        sidecar = path if path.suffix.lower() == ".txt" else path.with_suffix(".txt")
        if not sidecar.exists():
            raise RuntimeError(f"Missing transcript sidecar: {sidecar}")
        text = sidecar.read_text(encoding="utf-8").strip()
        if partial_callback and text:
            partial_callback(text)
        return TranscriptResult(
            text=text,
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


def _looks_like_browser_challenge(exc: Exception) -> bool:
    message = str(exc).lower()
    return "<html" in message and (
        "cf_chl" in message
        or "cloudflare" in message
        or "/cdn-cgi/challenge-platform" in message
    )


def _looks_like_missing_api_scope(exc: Exception) -> bool:
    message = str(exc).lower()
    return "missing scopes:" in message or "insufficient permissions" in message


def _plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _openai_segments(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    segments = raw.get("segments") or []
    normalized = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start
        normalized_segment = dict(segment)
        normalized_segment["text"] = text
        normalized_segment["start"] = start
        normalized_segment["end"] = end
        normalized.append(normalized_segment)
    return normalized


def _append_openai_segments(
    all_segments: list[dict[str, Any]],
    chunk_segments: list[dict[str, Any]],
    offset_seconds: float,
    chunk_duration_seconds: float | None = None,
) -> None:
    for raw_segment in chunk_segments:
        segment = dict(raw_segment)
        for key in ("start", "end"):
            try:
                value = float(segment.get(key, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if chunk_duration_seconds is not None:
                value = min(max(0.0, value), max(0.0, float(chunk_duration_seconds)))
            segment[key] = value + float(offset_seconds)
        if float(segment.get("end", 0.0)) < float(segment.get("start", 0.0)):
            segment["end"] = segment["start"]
        if all_segments:
            prev_text = str(all_segments[-1].get("text", ""))
            segment_text = str(segment.get("text", ""))
            if (
                prev_text
                and segment_text
                and not prev_text[-1].isspace()
                and not segment_text[0].isspace()
            ):
                segment["text"] = " " + segment_text
        all_segments.append(segment)


def _join_transcript_parts(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def _chunk_seconds(app_settings: PracticeSettings | None) -> int:
    if app_settings is None:
        return 60
    try:
        return max(10, int(app_settings.whisper_chunk_seconds))
    except (TypeError, ValueError):
        return 60


def _practice_settings() -> PracticeSettings | None:
    try:
        return PracticeSettings.load()
    except (OperationalError, ProgrammingError):
        return None


def _whisper_options(app_settings: PracticeSettings | None) -> dict[str, Any]:
    if app_settings is None:
        return {
            "beam_size": settings.WHISPER_BEAM_SIZE,
            "temperature": settings.WHISPER_TEMPERATURE,
            "condition_on_previous_text": settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            "no_speech_threshold": settings.WHISPER_NO_SPEECH_THRESHOLD,
            "_speech_practice_chunk_seconds": 60,
        }

    beam_size = int(app_settings.whisper_beam_size)
    temperature = float(app_settings.whisper_temperature)
    use_fp16 = False
    if app_settings.whisper_device == "gpu":
        use_fp16 = True
    elif app_settings.whisper_device == "auto":
        use_fp16 = _has_cuda()

    preset = app_settings.whisper_preset
    if preset == "fast_cpu":
        beam_size = 1
        temperature = 0.0
        use_fp16 = False
    elif preset == "balanced_cpu":
        beam_size = 2
        temperature = 0.0
        use_fp16 = False
    elif preset == "balanced_gpu":
        beam_size = 3
        temperature = 0.0
        use_fp16 = True
    elif preset == "accurate_gpu":
        beam_size = 5
        temperature = 0.0
        use_fp16 = True

    options: dict[str, Any] = {
        "temperature": temperature,
        "beam_size": beam_size,
        "condition_on_previous_text": bool(app_settings.whisper_condition_on_previous_text),
        "no_speech_threshold": float(app_settings.whisper_no_speech_threshold),
        "without_timestamps": not bool(app_settings.whisper_timestamps),
        "fp16": bool(use_fp16),
        "_speech_practice_chunk_seconds": int(app_settings.whisper_chunk_seconds),
    }
    if beam_size <= 1:
        options.pop("beam_size", None)
        options["best_of"] = 1
    return options


def clear_local_whisper_cache() -> None:
    _load_whisper_model.cache_clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
