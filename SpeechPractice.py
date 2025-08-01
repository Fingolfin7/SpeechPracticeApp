import sys
import os
import wave
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import whisper
from jiwer import wer

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

import db
from script_loader import pick_next_script


class ClickableViewBox(pg.ViewBox):
    """A ViewBox that calls a callback with the data-x on left-click."""
    def __init__(self, click_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.click_callback = click_callback

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self.click_callback:
            # Map scene-pos to data coords
            scene_pt = ev.scenePos()
            data_pt = self.mapSceneToView(scene_pt)
            self.click_callback(data_pt.x())
        ev.accept()


class SpeechPracticeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Clarity Practice")

        # Database, Whisper model, sampling rate
        self.db = db.get_session()
        self.sr = 16000
        self.model = whisper.load_model("base")

        # State containers
        self.audio_buffer = []
        self.audio_data = None
        self.current_audio_path = None
        self.is_recording = False
        self.play_start_ts = None
        self.play_start_index = 0

        self._build_ui()
        self._load_history()
        self.load_next_script()

    def _build_ui(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ── Left: Session history ──
        hist_w = QtWidgets.QWidget()
        h_l = QtWidgets.QVBoxLayout(hist_w)
        h_l.addWidget(QtWidgets.QLabel("Session History"))
        self.history_list = QtWidgets.QListWidget()
        h_l.addWidget(self.history_list)
        splitter.addWidget(hist_w)

        # ── Middle: Current script ──
        scr_w = QtWidgets.QWidget()
        s_l = QtWidgets.QVBoxLayout(scr_w)
        s_l.addWidget(QtWidgets.QLabel("Current Script"))
        self.script_txt = QtWidgets.QTextEdit(readOnly=True)
        s_l.addWidget(self.script_txt)
        splitter.addWidget(scr_w)

        # ── Right: Waveform / Controls / Transcript ──
        right_w = QtWidgets.QWidget()
        r_l = QtWidgets.QVBoxLayout(right_w)

        # PlotWidget with our ClickableViewBox
        vb = ClickableViewBox(click_callback=self._on_wave_click)
        self.plot = pg.PlotWidget(viewBox=vb)
        self.plot.setYRange(-1, 1)
        self.wave_line = self.plot.plot(pen="c")
        self.playhead = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen("r", width=2)
        )
        self.plot.addItem(self.playhead)
        r_l.addWidget(self.plot)

        # Timer to animate playhead
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(50)
        self.play_timer.timeout.connect(self._update_playhead)

        # Control buttons
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_play   = QtWidgets.QPushButton("Play")
        self.btn_score  = QtWidgets.QPushButton("Score")
        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_play.clicked.connect(self._play_audio)
        self.btn_score.clicked.connect(self._transcribe_and_score)
        for b in (self.btn_record, self.btn_play, self.btn_score):
            b.setEnabled(False)
            ctrl.addWidget(b)
        r_l.addLayout(ctrl)

        # Transcript & scores
        r_l.addWidget(QtWidgets.QLabel("Transcript & Scores"))
        self.transcript_txt = QtWidgets.QTextEdit(readOnly=True)
        r_l.addWidget(self.transcript_txt)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)

    def _load_history(self):
        self.history_list.clear()
        sessions = db.get_all_sessions(self.db)
        if not sessions:
            self.history_list.addItem("No sessions yet")
        else:
            for sess in sessions:
                label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.UserRole, sess.id)
                self.history_list.addItem(item)
        self.history_list.itemClicked.connect(self._load_session)

    def load_next_script(self):
        name, text = pick_next_script()
        self.current_script_name = name
        self.current_script_text = text
        self.script_txt.setText(f"{name}\n\n{text}")
        self.wave_line.clear()
        self.playhead.setPos(0)
        self.transcript_txt.clear()
        self.audio_data = None
        self.current_audio_path = None
        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.btn_score.setEnabled(False)

    def _load_session(self, item):
        sess_id = item.data(QtCore.Qt.UserRole)
        sess    = db.get_session_by_id(self.db, sess_id)

        # Script & transcript
        self.current_script_name = sess.script_name
        self.current_script_text = sess.script_text
        self.script_txt.setText(sess.script_text)
        out = (
            f"Transcript:\n{sess.transcript}\n\n"
            f"WER: {sess.wer:.2%}, Clarity: {sess.clarity:.2%}, "
            f"Score: {sess.score}/5"
        )
        self.transcript_txt.setText(out)

        # Load WAV
        wf = wave.open(sess.audio_path, "rb")
        frames = wf.readframes(wf.getnframes())
        data = (np.frombuffer(frames, np.int16).astype(np.float32)
                / 32767)
        self.audio_data = data
        self.current_audio_path = sess.audio_path

        # Plot & reset playhead
        self.wave_line.setData(data)
        self.playhead.setPos(0)

        # Enable play/record
        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(False)

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
            callback=lambda indata, *_: self.audio_buffer.append(indata.copy())
        )
        self.stream.start()

        # Live waveform timer
        self.rec_timer = QtCore.QTimer(self)
        self.rec_timer.setInterval(50)
        self.rec_timer.timeout.connect(self._update_waveform)
        self.rec_timer.start()

    def _stop_record(self):
        self.stream.stop()
        self.rec_timer.stop()
        self.is_recording = False
        self.btn_record.setText("Record")

        # Save WAV
        audio = np.concatenate(self.audio_buffer, axis=0).flatten()
        self.audio_data = audio
        os.makedirs("recordings", exist_ok=True)
        fname = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        path = os.path.join("recordings", fname)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes((audio * 32767).astype(np.int16))
        self.current_audio_path = path

        # Plot full waveform
        self.wave_line.setData(audio)

        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(True)

    def _update_waveform(self):
        if not self.audio_buffer: return
        data = np.concatenate(self.audio_buffer, axis=0).flatten()
        self.wave_line.setData(data[-self.sr :])

    def _play_audio(self):
        if self.audio_data is None: return
        sd.stop()
        self.play_start_index = 0
        self.play_start_ts = time.perf_counter()
        sd.play(self.audio_data, self.sr)
        self.play_timer.start()

    def _on_wave_click(self, x):
        """Seek & play from data-x (array index) on click."""
        if self.audio_data is None: return
        idx = int(x)
        idx = max(0, min(idx, len(self.audio_data) - 1))
        self.playhead.setPos(idx)
        sd.stop()
        self.play_start_index = idx
        self.play_start_ts = time.perf_counter()
        sd.play(self.audio_data[idx:], self.sr)
        self.play_timer.start()

    def _update_playhead(self):
        if self.audio_data is None or self.play_start_ts is None:
            return
        elapsed = time.perf_counter() - self.play_start_ts
        pos_idx = int(self.play_start_index + elapsed * self.sr)
        if pos_idx >= len(self.audio_data):
            self.play_timer.stop()
            return
        self.playhead.setPos(pos_idx)

    def _transcribe_and_score(self):
        self.transcript_txt.clear()
        res = self.model.transcribe(self.current_audio_path, fp16=False)
        hyp = res["text"].strip().lower()

        err = wer(self.current_script_text, hyp)
        clar = 1 - err
        score = round(clar * 4) + 1

        out = (
            f"Transcript:\n{hyp}\n\n"
            f"WER: {err:.2%}, Clarity: {clar:.2%}, "
            f"Score: {score}/5"
        )
        self.transcript_txt.setText(out)

        sess = db.add_session(
            self.db,
            self.current_script_name,
            self.current_script_text,
            self.current_audio_path,
            hyp, err, clar, score
        )
        label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.UserRole, sess.id)
        self.history_list.addItem(item)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SpeechPracticeApp()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()