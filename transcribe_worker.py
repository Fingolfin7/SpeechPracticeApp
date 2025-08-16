from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
from PyQt5 import QtCore
from jiwer import wer
import math
import re


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
            result = self._model.transcribe(self._audio_path, **self._options)
            hyp = self.clean_text(result["text"])
            err = wer(reference=self._ref, hypothesis=hyp)
            clar = 1.0 - err
            score = self._scale_score(clar)

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

    def run(self) -> None:
        try:
            result = self._model.transcribe(self._audio_source, **self._options)
            hyp = str(result["text"]).strip()
            segments = result.get("segments") if isinstance(result, dict) else None
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