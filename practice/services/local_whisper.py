from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

WHISPER_SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 60.0
CHUNK_SECONDS_OPTION = "_speech_practice_chunk_seconds"


def _chunk_seconds_from_env() -> float:
    try:
        return max(
            10.0,
            float(
                os.getenv(
                    "SPEECH_PRACTICE_TRANSCRIBE_CHUNK_SECONDS",
                    DEFAULT_CHUNK_SECONDS,
                )
            ),
        )
    except Exception:
        return DEFAULT_CHUNK_SECONDS


def _chunk_seconds_from_options(options: dict[str, Any]) -> float:
    raw_value = options.pop(CHUNK_SECONDS_OPTION, None)
    if raw_value is None:
        return _chunk_seconds_from_env()
    try:
        return max(10.0, float(raw_value))
    except Exception:
        return _chunk_seconds_from_env()


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    return np.ascontiguousarray(samples.reshape(-1))


def _resample_linear(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if int(source_sr) == int(target_sr):
        return _as_mono_float32(audio)

    samples = _as_mono_float32(audio)
    if samples.size == 0:
        return samples

    duration = samples.size / float(source_sr)
    target_size = max(1, int(round(duration * float(target_sr))))
    src_pos = np.linspace(
        0.0,
        max(0, samples.size - 1),
        num=samples.size,
        dtype=np.float64,
    )
    dst_pos = np.linspace(
        0.0,
        max(0, samples.size - 1),
        num=target_size,
        dtype=np.float64,
    )
    return np.interp(dst_pos, src_pos, samples).astype(np.float32, copy=False)


def _path_duration_seconds(path: str) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return None


def _audio_duration_seconds(audio_source: str | np.ndarray) -> float | None:
    if isinstance(audio_source, str):
        return _path_duration_seconds(audio_source)
    try:
        audio = _as_mono_float32(audio_source)
        return float(audio.size) / float(WHISPER_SAMPLE_RATE)
    except Exception:
        return None


def _iter_array_chunks(
    audio: np.ndarray,
    chunk_seconds: float,
) -> Iterable[tuple[np.ndarray, float]]:
    samples = _as_mono_float32(audio)
    chunk_samples = max(1, int(round(chunk_seconds * WHISPER_SAMPLE_RATE)))
    for start in range(0, samples.size, chunk_samples):
        chunk = samples[start : start + chunk_samples]
        if chunk.size:
            yield np.ascontiguousarray(chunk), float(start) / float(WHISPER_SAMPLE_RATE)


def _iter_file_chunks(
    audio_path: str,
    chunk_seconds: float,
) -> Iterable[tuple[np.ndarray, float]]:
    import soundfile as sf

    info = sf.info(audio_path)
    block_size = max(1, int(round(chunk_seconds * float(info.samplerate))))
    start_frame = 0
    with sf.SoundFile(audio_path, "r") as audio_file:
        while True:
            block = audio_file.read(block_size, dtype="float32", always_2d=False)
            if block is None or len(block) == 0:
                break
            offset = float(start_frame) / float(info.samplerate)
            start_frame += len(block)
            yield _resample_linear(block, info.samplerate, WHISPER_SAMPLE_RATE), offset


def _call_transcribe(
    model: Any,
    audio_source: str | np.ndarray,
    options: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        with torch.inference_mode():
            return model.transcribe(audio_source, **options)
    return model.transcribe(audio_source, **options)


def _segment_with_offset(segment: dict[str, Any], offset_seconds: float) -> dict[str, Any]:
    copied = dict(segment)
    for key in ("start", "end"):
        try:
            copied[key] = float(copied.get(key, 0.0)) + float(offset_seconds)
        except Exception:
            pass
    return copied


def _make_chunk_options(
    options: dict[str, Any],
    previous_text: str,
) -> dict[str, Any]:
    chunk_options = dict(options)
    if previous_text and bool(options.get("condition_on_previous_text", True)):
        existing_prompt = str(options.get("initial_prompt") or "").strip()
        prompt_tail = previous_text[-500:].strip()
        prompt = f"{existing_prompt} {prompt_tail}".strip()
        if prompt:
            chunk_options["initial_prompt"] = prompt
    return chunk_options


def _append_segments(
    all_segments: list[dict[str, Any]],
    chunk_segments: object,
    offset_seconds: float,
) -> None:
    if not isinstance(chunk_segments, list):
        return

    for raw_segment in chunk_segments:
        if not isinstance(raw_segment, dict):
            continue
        segment = _segment_with_offset(raw_segment, offset_seconds)
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


def transcribe_source(
    model: Any,
    audio_source: str | np.ndarray,
    options: dict[str, Any] | None = None,
    partial_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Transcribe in chunks for long audio so Whisper never has to hold the full
    decoded input at once. Small or non-streamable inputs keep the normal path.
    """
    opts = dict(options or {})
    chunk_seconds = _chunk_seconds_from_options(opts)
    duration_seconds = _audio_duration_seconds(audio_source)

    if duration_seconds is None or duration_seconds <= chunk_seconds:
        return _call_transcribe(model, audio_source, opts)

    if isinstance(audio_source, str):
        try:
            chunks = _iter_file_chunks(audio_source, chunk_seconds)
        except Exception:
            return _call_transcribe(model, audio_source, opts)
    else:
        chunks = _iter_array_chunks(audio_source, chunk_seconds)

    text_parts: list[str] = []
    all_segments: list[dict[str, Any]] = []
    last_result: dict[str, Any] = {}

    for chunk, offset_seconds in chunks:
        if chunk.size == 0:
            continue

        combined_text = _join_transcript_parts(text_parts)
        chunk_result = _call_transcribe(
            model,
            chunk,
            _make_chunk_options(opts, combined_text),
        )
        last_result = chunk_result if isinstance(chunk_result, dict) else {}
        chunk_text = str(last_result.get("text", "")).strip()
        if chunk_text:
            text_parts.append(chunk_text)
        _append_segments(all_segments, last_result.get("segments"), offset_seconds)
        if partial_callback is not None:
            try:
                partial_callback(_join_transcript_parts(text_parts))
            except Exception:
                pass

    stitched = dict(last_result)
    stitched["text"] = _join_transcript_parts(text_parts)
    if all_segments:
        stitched["segments"] = all_segments
    return stitched
