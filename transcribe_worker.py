from PyQt5 import QtCore
from jiwer import wer
import math


def _scale_score(clarity: float) -> float:
    # in `transcribe_worker.py`, replace the existing score calculation with:

    # Clamp clarity between 0 and 1
    clarity = max(0.0, min(clarity, 1.0))
    # Sigmoid centered at 0.75, steepness 20
    score = 1 + 4 / (1 + math.exp(-20 * (clarity - 0.75)))
    return min(5, max(1, score))


class TranscribeWorker(QtCore.QThread):
    """
    Runs Whisper in a background thread so the GUI stays responsive.
    Emits:
      completed(hyp: str, err: float, clar: float, score: float)
    """

    completed = QtCore.pyqtSignal(str, float, float, float)

    def __init__(
        self, model, ref_text: str, audio_path: str, parent=None
    ):
        super().__init__(parent)
        self._model = model
        self._ref = ref_text
        self._audio_path = audio_path

    def run(self) -> None:
        hyp = (
            self._model.transcribe(self._audio_path, fp16=False)["text"]
            .strip()
            .lower()
        )
        err = wer(self._ref, hyp)
        clar = 1.0 - err

        score = _scale_score(clar)

        self.completed.emit(hyp, err, clar, score)


class FreeTranscribeWorker(QtCore.QThread):
    """
    Runs Whisper in a background thread and emits only the transcript text.
    Emits:
      completed(hyp: str)
    """

    completed = QtCore.pyqtSignal(str)

    def __init__(self, model, audio_path: str, parent=None):
        super().__init__(parent)
        self._model = model
        self._audio_path = audio_path

    def run(self) -> None:
        hyp = (
            self._model.transcribe(self._audio_path, fp16=False)["text"]
            .strip()
        )
        self.completed.emit(hyp)