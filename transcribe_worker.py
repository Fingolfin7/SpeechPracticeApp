# ───── insert somewhere near the top of speech_practice.py ─────────────
from PyQt5 import QtCore
from jiwer import wer


class TranscribeWorker(QtCore.QThread):
    """
    Runs Whisper in a background thread so the GUI stays responsive.
    Emits:
      completed(hyp: str, err: float, clar: float, score: int)
    """

    completed = QtCore.pyqtSignal(str, float, float, int)

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
        score = round(clar * 4) + 1
        self.completed.emit(hyp, err, clar, score)