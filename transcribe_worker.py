from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from PyQt5 import QtCore
import math
import re
import time
from alignment_utils import compute_flexible_wer


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


def _chunk_seconds_from_options(options: Dict) -> float:
    raw_value = options.pop(CHUNK_SECONDS_OPTION, None)
    if raw_value is None:
        return _chunk_seconds_from_env()
    try:
        return max(10.0, float(raw_value))
    except Exception:
        return _chunk_seconds_from_env()


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1, dtype=np.float32)
    return np.ascontiguousarray(x.reshape(-1))


def _resample_linear(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if int(source_sr) == int(target_sr):
        return _as_mono_float32(audio)
    x = _as_mono_float32(audio)
    if x.size == 0:
        return x
    duration = x.size / float(source_sr)
    target_size = max(1, int(round(duration * float(target_sr))))
    src_pos = np.linspace(0.0, max(0, x.size - 1), num=x.size, dtype=np.float64)
    dst_pos = np.linspace(0.0, max(0, x.size - 1), num=target_size, dtype=np.float64)
    return np.interp(dst_pos, src_pos, x).astype(np.float32, copy=False)


def _path_duration_seconds(path: str) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return None


def _audio_duration_seconds(audio_source: Union[str, np.ndarray]) -> Optional[float]:
    if isinstance(audio_source, str):
        return _path_duration_seconds(audio_source)
    try:
        audio = _as_mono_float32(audio_source)
        return float(audio.size) / float(WHISPER_SAMPLE_RATE)
    except Exception:
        return None


def _iter_array_chunks(
    audio: np.ndarray, chunk_seconds: float
) -> Iterable[Tuple[np.ndarray, float]]:
    x = _as_mono_float32(audio)
    chunk_samples = max(1, int(round(chunk_seconds * WHISPER_SAMPLE_RATE)))
    for start in range(0, x.size, chunk_samples):
        chunk = x[start : start + chunk_samples]
        if chunk.size:
            yield np.ascontiguousarray(chunk), float(start) / float(WHISPER_SAMPLE_RATE)


def _iter_file_chunks(
    audio_path: str, chunk_seconds: float
) -> Iterable[Tuple[np.ndarray, float]]:
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


def _call_transcribe(model, audio_source: Union[str, np.ndarray], options: Dict) -> Dict:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        with torch.inference_mode():
            return model.transcribe(audio_source, **options)
    return model.transcribe(audio_source, **options)


def _segment_with_offset(segment: dict, offset_seconds: float) -> dict:
    seg = dict(segment)
    for key in ("start", "end"):
        try:
            seg[key] = float(seg.get(key, 0.0)) + float(offset_seconds)
        except Exception:
            pass
    return seg


def _make_chunk_options(options: Dict, previous_text: str) -> Dict:
    chunk_options = dict(options)
    if previous_text and bool(options.get("condition_on_previous_text", True)):
        existing_prompt = str(options.get("initial_prompt") or "").strip()
        prompt_tail = previous_text[-500:].strip()
        prompt = f"{existing_prompt} {prompt_tail}".strip()
        if prompt:
            chunk_options["initial_prompt"] = prompt
    return chunk_options


def _append_segments(
    all_segments: list[dict], chunk_segments: object, offset_seconds: float
) -> None:
    if not isinstance(chunk_segments, list):
        return
    for raw_seg in chunk_segments:
        if not isinstance(raw_seg, dict):
            continue
        seg = _segment_with_offset(raw_seg, offset_seconds)
        if all_segments:
            prev_text = str(all_segments[-1].get("text", ""))
            seg_text = str(seg.get("text", ""))
            if (
                prev_text
                and seg_text
                and not prev_text[-1].isspace()
                and not seg_text[0].isspace()
            ):
                seg["text"] = " " + seg_text
        all_segments.append(seg)


def transcribe_source(
    model,
    audio_source: Union[str, np.ndarray],
    options: Optional[Dict] = None,
    partial_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
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
    all_segments: list[dict] = []
    last_result: Dict = {}

    for chunk, offset_seconds in chunks:
        if chunk.size == 0:
            continue
        combined_text = " ".join(part.strip() for part in text_parts if part.strip())
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
                partial_callback(
                    " ".join(part.strip() for part in text_parts if part.strip())
                )
            except Exception:
                pass

    stitched = dict(last_result)
    stitched["text"] = " ".join(part.strip() for part in text_parts if part.strip())
    if all_segments:
        stitched["segments"] = all_segments
    return stitched


class TranscribeWorker(QtCore.QThread):
    """
    Runs Whisper in a background thread so the GUI stays responsive.
    Emits:
      completed(hyp: str, err: float, clar: float, score: float)
      completed_with_segments(hyp: str, err: float, clar: float, score: float, segments: object)
      failed(message: str)
    """

    completed = QtCore.pyqtSignal(str, float, float, float)
    completed_with_segments = QtCore.pyqtSignal(str, float, float, float, object)
    partial = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        model,
        ref_text: str,
        audio_path: str,
        parent=None,
        options: Optional[Dict] = None,
    ):
        super().__init__(parent)
        self._model = model
        self._ref = self.clean_text(ref_text)
        self._audio_path = audio_path
        self._options = options or {}
        self.last_timing: Dict[str, float] = {}

    @staticmethod
    def _scale_score(clarity: float) -> float:
        clarity = max(0.0, min(clarity, 1.0))
        score = 1 + 4 / (1 + math.exp(-20 * (clarity - 0.80)))
        return min(5, max(1, score))

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            result = transcribe_source(
                self._model,
                self._audio_path,
                self._options,
                partial_callback=self.partial.emit,
            )
            t_asr = time.perf_counter()
            hyp = self.clean_text(result["text"])
            err = compute_flexible_wer(self._ref, hyp)
            clar = 1.0 - err
            score = self._scale_score(clar)
            t_done = time.perf_counter()
            self.last_timing = {
                "asr_s": float(t_asr - t0),
                "worker_post_s": float(t_done - t_asr),
                "worker_total_s": float(t_done - t0),
            }

            segments = result.get("segments") if isinstance(result, dict) else None
            if segments is not None:
                try:
                    self.completed_with_segments.emit(hyp, err, clar, score, segments)
                except Exception:
                    pass
            self.completed.emit(hyp, err, clar, score)
        except Exception as e:
            try:
                self.failed.emit(str(e))
            except Exception:
                pass


class FreeTranscribeWorker(QtCore.QThread):
    """
    Runs Whisper in a background thread and emits only the transcript text.
    Emits:
      completed(hyp: str)
      completed_with_segments(hyp: str, segments: object)
      failed(message: str)
    """

    completed = QtCore.pyqtSignal(str)
    completed_with_segments = QtCore.pyqtSignal(str, object)
    partial = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        model,
        audio_source: Union[str, np.ndarray],
        parent=None,
        options: Optional[Dict] = None,
    ):
        super().__init__(parent)
        self._model = model
        self._audio_source: Union[str, np.ndarray] = audio_source
        self._options = options or {}
        self.last_timing: Dict[str, float] = {}

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            result = transcribe_source(
                self._model,
                self._audio_source,
                self._options,
                partial_callback=self.partial.emit,
            )
            t_asr = time.perf_counter()
            hyp = str(result["text"]).strip()
            segments = result.get("segments") if isinstance(result, dict) else None
            t_done = time.perf_counter()
            self.last_timing = {
                "asr_s": float(t_asr - t0),
                "worker_post_s": float(t_done - t_asr),
                "worker_total_s": float(t_done - t0),
            }
            if segments is not None:
                try:
                    self.completed_with_segments.emit(hyp, segments)
                except Exception:
                    pass
            self.completed.emit(hyp)
        except Exception as e:
            try:
                self.failed.emit(str(e))
            except Exception:
                pass
