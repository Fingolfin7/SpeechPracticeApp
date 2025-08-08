import sys
import os
from datetime import datetime
import soundfile as sf

import numpy as np
import sounddevice as sd
import whisper
from jiwer import wer

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QAction, QMenu, QStyle
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
import pyqtgraph as pg

import db
from script_loader import pick_next_script
from audio_player import AudioPlayer
from transcribe_worker import TranscribeWorker


class SpeechPracticeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Clarity Practice")

        # database / model / audio
        self.db = db.get_session()
        self.sr = 16_000
        self.model = whisper.load_model("base")

        # runtime state
        self.audio_buffer = []
        self.audio_data = None
        self.current_audio_path = None
        self.is_recording = False

        self.stream = None          # recording stream
        self.rec_timer = None       # live-view timer
        self.player = AudioPlayer(self.sr)

        self._build_ui()
        self._load_history()
        self.load_next_script()

    # ─────────────────────────────── UI ────────────────────────────────
    def _build_ui(self):
        mb = self.menuBar()

        # next-script action directly on the menubar
        act_next = mb.addAction("Next Script")
        act_next.triggered.connect(self.load_next_script)

        # optional File menu stub (future actions go here)
        mb.addMenu("File")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        # left ── session history
        hist_w = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(hist_w)
        hl.addWidget(QtWidgets.QLabel("Session History"))
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(
            self._history_context_menu
        )
        hl.addWidget(self.history_list)
        splitter.addWidget(hist_w)

        # middle ── current script
        scr_w = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(scr_w)
        sl.addWidget(QtWidgets.QLabel("Current Script"))
        self.script_txt = QtWidgets.QTextEdit(readOnly=True)
        sl.addWidget(self.script_txt)
        splitter.addWidget(scr_w)

        # right ── waveform / controls / transcript
        right_w = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right_w)

        # waveform plot (x-axis in seconds)
        vb = pg.ViewBox()
        self.plot = pg.PlotWidget(viewBox=vb)
        self.plot.setYRange(-1, 1)
        self.plot.getPlotItem().setLabel("bottom", "Time (s)")
        self.wave_line = self.plot.plot(pen="c")
        self.playhead = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen("r", width=2)
        )
        self.plot.addItem(self.playhead)
        vb.mouseClickEvent = self._vb_click_event
        rl.addWidget(self.plot)

        # timer to animate play-head
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(50)
        self.play_timer.timeout.connect(self._update_playhead)

        # record / play / score buttons
        ctrl = QtWidgets.QHBoxLayout()

        # icons
        self.ic_play = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.ic_pause = self.style().standardIcon(QStyle.SP_MediaPause)
        self.ic_stop = self.style().standardIcon(QStyle.SP_MediaStop)
        self.ic_record = self._make_record_icon()

        # buttons
        self.btn_record = QtWidgets.QPushButton()
        self.btn_record.setIcon(self.ic_record)
        self.btn_play = QtWidgets.QPushButton()
        self.btn_play.setIcon(self.ic_play)
        self.btn_score = QtWidgets.QPushButton("Score")

        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_score.clicked.connect(self._transcribe_and_score)

        for b in (self.btn_record, self.btn_play, self.btn_score):
            b.setEnabled(False)
            ctrl.addWidget(b)

        rl.addLayout(ctrl)

        # combined metrics label
        self.metrics_label = QtWidgets.QLabel(
            "Score: –, WER: –, Clarity: –"
        )
        f = self.metrics_label.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.metrics_label.setFont(f)
        rl.addWidget(self.metrics_label)

        # transcript pane
        rl.addWidget(QtWidgets.QLabel("Transcript"))
        self.transcript_txt = QtWidgets.QTextEdit(readOnly=True)
        rl.addWidget(self.transcript_txt)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)

    def _make_record_icon(self, size: int = 20) -> QIcon:
        pix = QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("red"))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(0, 0, size, size)
        p.end()
        return QIcon(pix)

    # ────────────────────── history list helpers ───────────────────────
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

    def _history_context_menu(self, pt):
        item = self.history_list.itemAt(pt)
        if not item or not isinstance(item.data(QtCore.Qt.UserRole), int):
            return
        menu = QMenu()
        act = QAction("Delete Session", self)
        act.triggered.connect(lambda: self._delete_session(item))
        menu.addAction(act)
        menu.exec_(self.history_list.mapToGlobal(pt))

    def _delete_session(self, item):
        sid = item.data(QtCore.Qt.UserRole)
        db.delete_session(self.db, sid)
        row = self.history_list.row(item)
        self.history_list.takeItem(row)
        if getattr(self, "current_session_id", None) == sid:
            self.load_next_script()

    # ───────────────────────── script handling ─────────────────────────
    def load_next_script(self):
        name, text = pick_next_script()
        self.current_script_name = name
        self.current_script_text = text
        self.script_txt.setText(f"{name}\n\n{text}")

        self.player.stop()
        self.play_timer.stop()
        self.wave_line.clear()
        self.playhead.setPos(0)
        self.metrics_label.setText("Score: –, WER: –, Clarity: –")
        self.transcript_txt.clear()
        self.audio_data = None
        self.current_audio_path = None

        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.btn_score.setEnabled(False)

    def _load_session(self, item):
        """
        Populate the UI with the chosen history entry and prepare a *fresh*
        AudioPlayer so that playback and seek work reliably after recording.
        """
        sid = item.data(QtCore.Qt.UserRole)
        sess = db.get_session_by_id(self.db, sid)
        self.current_session_id = sid

        # ── 1. close any existing player / play-timer ──────────────────────
        if hasattr(self, "player") and self.player is not None:
            try:
                self.player.close()
            except Exception:
                pass
        self.play_timer.stop()

        # ── 2. copy script and metrics to the widgets ──────────────────────
        self.current_script_name = sess.script_name
        self.current_script_text = sess.script_text
        self.script_txt.setText(sess.script_text)

        self.metrics_label.setText(
            f"Score: {sess.score:.2f}/5 | "
            f"WER: {sess.wer:.2%} | Clarity: {sess.clarity:.2%}"
        )
        self.transcript_txt.setText(sess.transcript)

        # ── 3. read the audio file and plot it (x-axis = seconds) ──────────
        data, file_sr = sf.read(sess.audio_path, dtype="float32")
        if hasattr(data, "ndim") and data.ndim > 1:
            data = data[:, 0]
        if file_sr != self.sr and data.size > 0:
            dur = data.size / float(file_sr)
            new_len = int(round(dur * self.sr))
            x_old = np.linspace(0.0, dur, num=data.size, endpoint=False)
            x_new = np.linspace(0.0, dur, num=new_len, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)

        self.audio_data = data
        self.current_audio_path = sess.audio_path

        x = np.arange(data.size) / self.sr
        self.wave_line.setData(x, data)
        self.playhead.setPos(0)

        # ── 4. create a *new* AudioPlayer and load the clip ────────────────
        self.player = AudioPlayer(self.sr)
        self.player.set_data(data)

        # ── 5. enable the appropriate buttons ──────────────────────────────
        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(False)

    # ────────────────────────── recording ──────────────────────────────
    def _toggle_record(self):
        if not self.is_recording:
            self._start_record()
            self.btn_record.setIcon(self.ic_stop)  # recording → stop
        else:
            self._stop_record()
            self.btn_record.setIcon(self.ic_record)  # back to record

    def _start_record(self):
        self.play_timer.stop()
        self.player.close()

        self.audio_buffer = []
        self.is_recording = True
        self.btn_record.setText("Stop")
        self.btn_play.setEnabled(False)
        self.btn_score.setEnabled(False)

        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            callback=lambda indata, *_: self.audio_buffer.append(
                indata.copy()
            ),
        )
        self.stream.start()

        self.rec_timer = QtCore.QTimer(self)
        self.rec_timer.setInterval(50)
        self.rec_timer.timeout.connect(self._update_waveform)
        self.rec_timer.start()

    def _stop_record(self):
        """
        Finish the current recording, save the clip, update the UI and
        create a fresh AudioPlayer so that playback continues to work.
        """
        # ── 1. stop / close the input stream and live-view timer ─────────────────
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.rec_timer is not None:
            self.rec_timer.stop()
            self.rec_timer = None

        self.is_recording = False
        self.btn_record.setText("Record")

        # ── 2. concatenate the buffered blocks and trim leading / trailing silence
        if not self.audio_buffer:
            return  # nothing was recorded

        raw = np.concatenate(self.audio_buffer, axis=0).flatten()
        trimmed = self._trim_silence(raw)
        if trimmed.size < self.sr // 10:
            trimmed = raw  # clip is too short after trimming

        self.audio_data = trimmed

        # ── 3. write the FLAC file to recordings/ ──────────────────────────
        os.makedirs("recordings", exist_ok=True)
        fname = datetime.now().strftime("%Y%m%d_%H%M%S") + ".flac"
        path = os.path.join("recordings", fname)
        sf.write(path, trimmed.astype(np.float32), self.sr, format="FLAC", subtype="PCM_16")
        self.current_audio_path = path

        # ── 4. update waveform plot (x-axis in seconds) ────────────────────
        x = np.arange(trimmed.size) / self.sr
        self.wave_line.setData(x, trimmed)
        self.playhead.setPos(0)

        # ── 5. create a brand-new AudioPlayer and load the clip ────────────
        self.player = AudioPlayer(self.sr)
        self.player.set_data(trimmed)

        # ── 6. re-enable the play / score buttons ──────────────────────────
        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(True)

    def _trim_silence(self, data, thresh=0.02):
        mask = np.abs(data) > thresh
        if not mask.any():
            return data
        i1 = np.argmax(mask)
        i2 = len(mask) - np.argmax(mask[::-1])
        return data[i1:i2]

    def _update_waveform(self):
        if not self.audio_buffer:
            return
        snippet = np.concatenate(self.audio_buffer, axis=0).flatten()
        snippet = snippet[-self.sr :]
        x = np.arange(snippet.size) / self.sr
        self.wave_line.setData(x, snippet)

    # ───────────────────────── playback ────────────────────────────────
    def _toggle_play_pause(self):
        """
        Play, resume or pause depending on current state.
        """
        if self.audio_data is None:
            return

        if self.player.active:
            # ⇒ pause
            self.player.pause()
            self.play_timer.stop()
            self.btn_play.setIcon(self.ic_play)
            return

        # ⇒ (re-)start
        start = (
            self.player.idx
            if 0 <= self.player.idx < self.audio_data.size
            else 0
        )
        self.player.set_data(self.audio_data)
        self.player.play(start)
        self.play_timer.start()
        self.btn_play.setIcon(self.ic_pause)

    def _vb_click_event(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or self.audio_data is None:
            ev.ignore()
            return
        pos = ev.scenePos()
        vb = self.plot.getPlotItem().getViewBox()
        t = vb.mapSceneToView(pos).x()            # seconds
        t = max(0.0, min(t, len(self.audio_data) / self.sr))
        idx = int(t * self.sr)

        self.playhead.setPos(t)
        self.player.set_data(self.audio_data)
        self.player.play(idx)
        self.play_timer.start()
        ev.accept()

    def _update_playhead(self):
        if not self.player.active:
            self.play_timer.stop()
            self.btn_play.setIcon(self.ic_play)  # ← new line
            return
        self.playhead.setPos(self.player.idx / self.sr)

    # ─────────────────────── scoring / storage ─────────────────────────
    # ───── inside class SpeechPracticeApp – drop-in replacements ───────────
    def _transcribe_and_score(self) -> None:
        """
        Start a background transcription task and disable the button until
        the job is finished.
        """
        if self.current_audio_path is None:
            return

        self.btn_score.setEnabled(False)
        self.metrics_label.setText("Scoring… please wait")

        self.worker = TranscribeWorker(
            self.model,
            self.current_script_text,
            self.current_audio_path,
            parent=self,  # same thing, keyword form is clearer
        )
        self.worker.completed.connect(self._on_transcription_done)
        self.worker.start()

    @QtCore.pyqtSlot(str, float, float, float)
    def _on_transcription_done(
            self, hyp: str, err: float, clar: float, score: float
    ) -> None:
        """
        Slot called when the transcription thread finishes.
        Updates UI and stores the session in the DB.
        """
        # update UI
        self.metrics_label.setText(
            f"Score: {score:.2f}/5 | WER: {err:.2%} | Clarity: {clar:.2%}"
        )
        self.transcript_txt.setText(hyp)
        self.btn_score.setEnabled(True)

        # persist to DB
        sess = db.add_session(
            self.db,
            self.current_script_name,
            self.current_script_text,
            self.current_audio_path,
            hyp,
            err,
            clar,
            score,
        )
        label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.UserRole, sess.id)
        self.history_list.addItem(item)


# ────────────────────────── main entry ────────────────────────────────
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SpeechPracticeApp()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()