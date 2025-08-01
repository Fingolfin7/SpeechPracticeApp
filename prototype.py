import sys
import os
import json
import random
import wave
import threading

import numpy as np
import sounddevice as sd
import whisper
from jiwer import wer
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

def pick_next_script():
    """Round-robin+shuffle picker from scripts/*.txt."""
    SCRIPTS_DIR = r"scripts"
    INDEX_FILE  = "script_index.json"
    files = sorted(f for f in os.listdir(SCRIPTS_DIR)
                   if f.lower().endswith(".txt"))
    if os.path.exists(INDEX_FILE):
        idx = json.load(open(INDEX_FILE))
    else:
        idx = {"pos": 0, "order": []}
    if len(idx.get("order", [])) != len(files):
        idx["order"] = list(range(len(files)))
        random.shuffle(idx["order"])
        idx["pos"]   = 0
    i = idx["order"][idx["pos"]]
    idx["pos"] = (idx["pos"] + 1) % len(files)
    json.dump(idx, open(INDEX_FILE, "w"), indent=2)
    path = os.path.join(SCRIPTS_DIR, files[i])
    text = open(path, encoding="utf-8").read().strip().lower()
    return files[i], text

class SpeechPracticeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Clarity Practice")
        self.sr   = 16000
        self.model = whisper.load_model("base")
        self.audio_buffer = []
        self.is_recording = False
        self._build_ui()
        self.load_next_script()

    def _build_ui(self):
        ctr = QtWidgets.QWidget()
        self.setCentralWidget(ctr)
        layout = QtWidgets.QVBoxLayout(ctr)

        # 1) Script display
        self.script_txt = QtWidgets.QTextEdit(readOnly=True)
        self.script_txt.setFixedHeight(100)
        layout.addWidget(self.script_txt)

        # 2) Controls
        hl = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_record.clicked.connect(self._toggle_record)
        hl.addWidget(self.btn_record)
        self.btn_play   = QtWidgets.QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._playback)
        hl.addWidget(self.btn_play)
        self.btn_score  = QtWidgets.QPushButton("Transcribe & Score")
        self.btn_score.setEnabled(False)
        self.btn_score.clicked.connect(self._transcribe_score)
        hl.addWidget(self.btn_score)
        layout.addLayout(hl)

        # 3) Waveform
        self.plot = pg.PlotWidget()
        self.plot.setYRange(-1,1)
        self.wave_line = self.plot.plot(pen="c")
        layout.addWidget(self.plot)

        # 4) Results
        self.result_txt = QtWidgets.QTextEdit(readOnly=True)
        self.result_txt.setFixedHeight(100)
        layout.addWidget(self.result_txt)

        # Menu for next script
        mb = self.menuBar()
        fm = mb.addMenu("File")
        na = QtWidgets.QAction("Next Script", self)
        na.triggered.connect(self.load_next_script)
        fm.addAction(na)

    def load_next_script(self):
        name, txt = pick_next_script()
        self.ref_script = txt
        self.script_txt.setText(f"{name}\n\n{txt}")

    def _toggle_record(self):
        if not self.is_recording:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        self.audio_buffer = []
        self.is_recording = True
        self.btn_record.setText("Stop")
        self.btn_play.setEnabled(False)
        self.btn_score.setEnabled(False)
        self.stream = sd.InputStream(
            samplerate=self.sr, channels=1,
            callback=lambda indata,frm,tm,st:
                      self.audio_buffer.append(indata.copy())
        )
        self.stream.start()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_waveform)
        self.timer.start(50)

    def _stop_record(self):
        self.stream.stop()
        self.timer.stop()
        self.is_recording = False
        self.btn_record.setText("Record")
        audio = np.concatenate(self.audio_buffer, axis=0).flatten()
        self.audio_data = audio
        # save WAV
        with wave.open("practice.wav","wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes((audio*32767).astype(np.int16))
        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(True)

    def _update_waveform(self):
        if self.audio_buffer:
            data = np.concatenate(self.audio_buffer,0).flatten()
            data = data[-self.sr:]  # last second
            self.wave_line.setData(data)

    def _playback(self):
        sd.play(self.audio_data, self.sr)

    def _transcribe_score(self):
        self.result_txt.clear()
        # 1) Transcribe
        res = self.model.transcribe("practice.wav", fp16=False)
        hyp = res["text"].strip().lower()
        # 2) Compute WER & clarity
        err = wer(self.ref_script, hyp)
        clar = 1 - err
        score = round(clar*4) + 1
        # 3) Show results
        out = (
            f"Transcript:\n{hyp}\n\n"
            f"WER: {err:.2%}, Clarity: {clar:.2%}, "
            f"Score: {score}/5"
        )
        self.result_txt.setText(out)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SpeechPracticeApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()