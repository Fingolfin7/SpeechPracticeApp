# ───── insert somewhere near the top of speech_practice.py ─────────────
from PyQt5 import QtCore
from jiwer import wer
import math


def _scale_score(clar: float) -> float:
    # in `transcribe_worker.py`, replace the existing score calculation with:

    if clar < 0.5:
        # clarity 0–50 % → score 1–2
        score = max(1, round(clar / 0.5 * 2))
    elif clar < 0.8:
        # clarity 50 %–80 % → score 2–4
        score = round((clar - 0.5) / 0.3 * 2 + 2)
    else:
        # clarity 80 %–100 % → score 4–100
        score = round((clar - 0.8) / 0.2 * 96 + 4)

    return score


class TranscribeWorker(QtCore.QThread):
    """
    Runs Whisper in a background thread so the GUI stays responsive.
    Emits:
      completed(hyp: str, err: float, clar: float, score: float)
    """

    completed = QtCore.pyqtSignal(str, float, float, float)

    def __init__(
        self, model, ref_text: str, wav_path: str, parent=None
    ):
        super().__init__(parent)
        self._model = model
        self._ref = ref_text
        self._wav = wav_path

    def run(self) -> None:
        hyp = (
            self._model.transcribe(self._wav, fp16=False)["text"]
            .strip()
            .lower()
        )
        err = wer(self._ref, hyp)
        clar = 1.0 - err

        score = _scale_score(clar)

        self.completed.emit(hyp, err, clar, score)