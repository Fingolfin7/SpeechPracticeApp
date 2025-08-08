# speech_practice.py
# prettier-ignore
from __future__ import annotations

import os
import sys
from datetime import datetime
import soundfile as sf

import numpy as np
import sounddevice as sd
import whisper
from jiwer import wer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import db
from audio_player import AudioPlayer
from script_loader import pick_next_script
from transcribe_worker import TranscribeWorker

# ───────────────────────────── theming ───────────────────────────────────

def apply_modern_theme(app: QtWidgets.QApplication) -> None:
    """Apply a clean, modern dark theme with a cyan accent."""
    app.setStyle("Fusion")

    # Base font
    try:
        app.setFont(QtGui.QFont("Segoe UI", 10))
    except Exception:
        pass

    # Dark palette
    bg = QtGui.QColor("#0f141a")
    panel = QtGui.QColor("#151b22")
    text = QtGui.QColor("#e6eaf0")
    accent = QtGui.QColor("#00d0ff")

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, bg)
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, panel)
    pal.setColor(QtGui.QPalette.AlternateBase, bg)
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, panel)
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.ToolTipBase, panel)
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    pal.setColor(QtGui.QPalette.Highlight, accent)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#0f141a"))
    app.setPalette(pal)

    # App-wide stylesheet (widgets keep subtle borders/padding)
    app.setStyleSheet(
        """
QMainWindow{background:#0f141a;}
QMenuBar{background:#0f141a;color:#e6eaf0;}
QMenuBar::item:selected{background:#151b22;}
QMenu{background:#151b22;color:#e6eaf0;border:1px solid #202833;}
QMenu::item:selected{background:#1f2a36;}
QSplitter::handle{background:#0f141a;width:6px;margin:0 4px;}
QSplitter::handle:hover{background:#1a2230;}
QLabel{color:#e6eaf0;}
QListWidget{background:#151b22;color:#e6eaf0;border:1px solid #202833;border-radius:6px;}
QListWidget::item:selected{background:#0e639c;}
QTextEdit{background:#151b22;color:#e6eaf0;border:1px solid #202833;border-radius:6px;}
QPushButton{background:#1d2633;color:#e6eaf0;border:1px solid #263241;border-radius:8px;padding:8px 14px;}
QPushButton:hover{background:#223043;}
QPushButton:disabled{color:#6f7c91;border-color:#2b3747;}
QPushButton#PrimaryButton{background:#00d0ff;color:#0f141a;font-weight:600;border:0;padding:8px 18px;border-radius:18px;}
QPushButton#PrimaryButton:hover{background:#5ee0ff;}
QPushButton#RecordBtn{background:#e5484d;border:0;width:44px;height:44px;border-radius:22px;color:#0f141a;}
QPushButton#RecordBtn:pressed{background:#ff5b61;}
QPushButton#CircleBtn{background:#1d2633;width:44px;height:44px;border-radius:22px;}
QWidget#Transport{background:#151b22;border:1px solid #263241;border-radius:28px;}
QToolTip{background-color:#151b22;color:#e6eaf0;border:1px solid #202833;}
        """
    )

# ───────────────────────────── main window ────────────────────────────────


class SpeechPracticeApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speech Clarity Practice")

        # back-end
        self.db = db.get_session()
        self.sr = 16_000
        self.model = whisper.load_model("base")

        # state
        self.audio_buffer: list[np.ndarray] = []
        self.audio_data: np.ndarray | None = None
        self.current_audio_path: str | None = None
        self.is_recording = False

        self.stream = None
        self.rec_timer: QtCore.QTimer | None = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(50)
        self.play_timer.timeout.connect(self._update_playhead)

        self.player = AudioPlayer(self.sr)

        self._build_ui()
        self._load_history()
        self.load_next_script()

    # ───────────────────────────────── UI ─────────────────────────────────

    def _build_ui(self) -> None:
        mb = self.menuBar()

        act_next = mb.addAction("Next Script")
        act_next.triggered.connect(self.load_next_script)
        # simple View menu for toggles
        view_menu = mb.addMenu("View")

        splitter = QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        # left: history
        hist_w = QWidget()
        hl = QVBoxLayout(hist_w)
        hl.addWidget(QLabel("Session History"))
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(
            self._history_context_menu
        )
        hl.addWidget(self.history_list)
        splitter.addWidget(hist_w)

        # middle: current script
        scr_w = QWidget()
        sl = QVBoxLayout(scr_w)
        sl.addWidget(QLabel("Current Script"))
        self.script_txt = QTextEdit(readOnly=True)
        sl.addWidget(self.script_txt)
        splitter.addWidget(scr_w)

        # right: waveform / controls / transcript
        right_w = QWidget()
        rl = QVBoxLayout(right_w)

        # waveform plot ----------------------------------------------------
        import pyqtgraph as pg

        vb = pg.ViewBox()
        self.plot = pg.PlotWidget(viewBox=vb)
        self.plot.setBackground(pg.mkColor("#0e1d1d"))
        self.plot.setMouseEnabled(y=False)
        self.plot.setMenuEnabled(False)
        self.plot.showAxis("left", False)
        self.plot.showAxis("right", False)
        self.plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.plot.setFrameStyle(pg.QtWidgets.QFrame.NoFrame)
        self.plot.showGrid(x=True, y=False, alpha=0.2)

        pen = pg.mkPen("#00d0ff", width=1.8)
        brush = pg.mkBrush("#00d0ff22")
        self.wave_line = self.plot.plot(fillLevel=0, pen=pen, brush=brush)

        self.playhead = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen("#e5484d", width=2)
        )
        self.plot.addItem(self.playhead)

        vb.mouseClickEvent = self._vb_click_event
        rl.addWidget(self.plot)

        # top-axis for time -------------------------------------------------
        self.top_axis = self.plot.getPlotItem().getAxis("top")
        self.top_axis.setHeight(20)
        self.plot.showAxis("bottom", False)
        self.plot.showAxis("top", True)
        try:
            self.top_axis.setPen(pg.mkPen("#2b3747"))
            self.top_axis.setTextPen(pg.mkPen("#6f7c91"))
        except Exception:
            pass

        # centred transport bar -------------------------------------------
        transport = QWidget(objectName="Transport")
        transport.setFixedHeight(56)
        tlay = QHBoxLayout(transport)
        tlay.setContentsMargins(10, 6, 10, 6)
        tlay.setSpacing(10)

        # icons (high-contrast glyphs)
        self.ic_play = self._make_play_icon()
        self.ic_pause = self._make_pause_icon()
        self.ic_stop = self._make_stop_icon()
        self.ic_record = self._make_record_icon()

        # buttons
        self.btn_record = QPushButton(objectName="RecordBtn")
        self.btn_record.setIcon(self.ic_record)
        self.btn_play = QPushButton(objectName="CircleBtn")
        self.btn_play.setIcon(self.ic_play)
        self.btn_score = QPushButton("Score")
        self.btn_score.setObjectName("PrimaryButton")
        # volume slider
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # 0..120 maps to ~0%..200% with perceptual curve
        self.vol_slider.setRange(0, 120)
        self.vol_slider.setValue(90)
        self.vol_slider.setFixedWidth(120)
        self.vol_slider.setToolTip("Volume")

        icon_sz = 28  # <— pick any size you like (px)

        # remake glyphs at the chosen size
        self.ic_record = self._make_record_icon(icon_sz)
        self.ic_play = self._make_play_icon(icon_sz)
        self.ic_pause = self._make_pause_icon(icon_sz)
        self.ic_stop = self._make_stop_icon(icon_sz)

        # enlarge the icons that live on the buttons
        self.btn_play.setIconSize(QtCore.QSize(icon_sz, icon_sz))
        self.btn_record.setIconSize(QtCore.QSize(icon_sz, icon_sz))

        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_score.clicked.connect(self._transcribe_and_score)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)

        for b in (self.btn_record, self.btn_play, self.btn_score):
            b.setEnabled(False)
            tlay.addWidget(b)
        tlay.addWidget(self.vol_slider)

        rl.addWidget(transport, alignment=QtCore.Qt.AlignHCenter)

        # metrics label ----------------------------------------------------
        self.metrics_label = QLabel("Score: –, WER: –, Clarity: –")
        self.metrics_label.setTextInteractionFlags( # enable copy-paste
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )
        f = self.metrics_label.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.metrics_label.setFont(f)
        rl.addWidget(self.metrics_label)

        # transcript pane --------------------------------------------------
        rl.addWidget(QLabel("Transcript"))
        self.transcript_txt = QTextEdit(readOnly=True)
        rl.addWidget(self.transcript_txt)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)

    # --------------------------- helpers ---------------------------------

    def _make_record_icon(self, size: int = 20) -> QtGui.QIcon:
        pix = QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        # High-contrast red circle with a dark outline
        p.setBrush(QColor("#ff4d57"))
        p.setPen(QColor("#0f141a"))
        p.drawEllipse(0, 0, size, size)
        p.end()
        return QtGui.QIcon(pix)

    def _make_play_icon(self, size: int = 20) -> QtGui.QIcon:
        w = h = size
        pix = QPixmap(w, h)
        pix.fill(QtCore.Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        # circular dark background + cyan triangle for contrast
        p.setBrush(QColor("#1d2633"))
        p.setPen(QColor("#0f141a"))
        p.drawEllipse(0, 0, w, h)
        p.setBrush(QColor("#e6eaf0"))
        p.setPen(QtCore.Qt.NoPen)
        # triangle
        margin = int(size * 0.28)
        points = [
            QtCore.QPoint(margin, margin),
            QtCore.QPoint(w - margin, h // 2),
            QtCore.QPoint(margin, h - margin),
        ]
        p.drawPolygon(QtGui.QPolygon(points))
        p.end()
        return QtGui.QIcon(pix)

    def _make_pause_icon(self, size: int = 20) -> QtGui.QIcon:
        w = h = size
        pix = QPixmap(w, h)
        pix.fill(QtCore.Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#1d2633"))
        p.setPen(QColor("#0f141a"))
        p.drawEllipse(0, 0, w, h)
        # pause bars
        bar_w = max(2, int(size * 0.18))
        gap = int(size * 0.14)
        x1 = w // 2 - gap - bar_w
        x2 = w // 2 + gap
        y = int(size * 0.24)
        bar_h = h - 2 * y
        p.setBrush(QColor("#e6eaf0"))
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(x1, y, bar_w, bar_h, 2, 2)
        p.drawRoundedRect(x2, y, bar_w, bar_h, 2, 2)
        p.end()
        return QtGui.QIcon(pix)

    def _make_stop_icon(self, size: int = 20) -> QtGui.QIcon:
        w = h = size
        pix = QPixmap(w, h)
        pix.fill(QtCore.Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#1d2633"))
        p.setPen(QColor("#0f141a"))
        p.drawEllipse(0, 0, w, h)
        p.setBrush(QColor("#e6eaf0"))
        p.setPen(QtCore.Qt.NoPen)
        s = int(size * 0.42)
        x = (w - s) // 2
        y = (h - s) // 2
        p.drawRoundedRect(x, y, s, s, 2, 2)
        p.end()
        return QtGui.QIcon(pix)

    # -------------------------- input hooks ------------------------------
    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        # Spacebar toggles play/pause when not typing in a text box
        if ev.key() == QtCore.Qt.Key_Space and not (self.script_txt.hasFocus() or self.transcript_txt.hasFocus()):
            self._toggle_play_pause()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _on_volume_changed(self, value: int) -> None:
        # Perceptual mapping: slider 0..120 -> gain 0.0..2.0
        # Below 100: approximate -inf..0 dB; Above 100: up to +6 dB with soft limiting
        v = max(0, min(int(value), 120))
        if v == 0:
            gain = 0.0
        elif v <= 100:
            # Map to -48..0 dB, then to linear
            db = (v - 100) * 0.48  # -48 dB at v=0, 0 dB at v=100
            gain = 10 ** (db / 20.0)
        else:
            # 100..120 -> 0..+6 dB (1.0..~2.0)
            db = (v - 100) * 0.3  # +6 dB at 120
            gain = 10 ** (db / 20.0)
        self.player.set_volume(gain)

    def _history_context_menu(self, pt) -> None:
        item = self.history_list.itemAt(pt)
        if not item or not isinstance(item.data(QtCore.Qt.UserRole), int):
            return
        menu = QMenu()
        act = QAction("Delete Session", self)
        act.triggered.connect(lambda: self._delete_session(item))
        menu.addAction(act)
        menu.exec_(self.history_list.mapToGlobal(pt))

    # --------------------------- data ------------------------------------

    def _load_history(self) -> None:
        self.history_list.clear()
        sessions = db.get_all_sessions(self.db)
        if not sessions:
            self.history_list.addItem("No sessions yet")
        else:
            for sess in sessions:
                label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
                it = QListWidgetItem(label)
                it.setData(QtCore.Qt.UserRole, sess.id)
                self.history_list.addItem(it)
        self.history_list.itemClicked.connect(self._load_session)

    def _delete_session(self, item) -> None:
        sid = item.data(QtCore.Qt.UserRole)
        db.delete_session(self.db, sid)
        row = self.history_list.row(item)
        self.history_list.takeItem(row)
        if getattr(self, "current_session_id", None) == sid:
            self.load_next_script()

    # --------------------------- script -----------------------------------

    def load_next_script(self) -> None:
        name, text = pick_next_script()
        self.current_script_name = name
        self.current_script_text = text
        # clear any active session id when switching scripts
        self.current_session_id = None
        self.script_txt.setText(f"{name}\n\n{text}")

        self.player.stop()
        self.play_timer.stop()
        self.wave_line.clear()
        self.playhead.setPos(0)
        self._update_time_axis(0)

        self.metrics_label.setText("Score: –, WER: –, Clarity: –")
        self.transcript_txt.clear()
        self.audio_data = None
        self.current_audio_path = None

        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.btn_score.setEnabled(False)

    # --------------------------- recording --------------------------------

    def _toggle_record(self) -> None:
        if not self.is_recording:
            self._start_record()
            self.btn_record.setIcon(self.ic_stop)
        else:
            self._stop_record()
            self.btn_record.setIcon(self.ic_record)

    def _start_record(self) -> None:
        self.player.pause()
        self.play_timer.stop()

        self.audio_buffer.clear()
        self.is_recording = True

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

        self.btn_play.setEnabled(False)
        self.btn_score.setEnabled(False)

    def _stop_record(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.rec_timer is not None:
            self.rec_timer.stop()
            self.rec_timer = None

        self.is_recording = False

        # Ensure we have something to save even if recording was too short to capture
        if not self.audio_buffer:
            # create a brief silent clip (0.5s) so a session can still be saved
            trimmed = np.zeros(int(self.sr * 0.5), dtype=np.float32)
        else:
            raw = np.concatenate(self.audio_buffer, axis=0).flatten()
            trimmed = self._trim_silence(raw)
            if trimmed.size < self.sr // 10:
                trimmed = raw
        self.audio_data = trimmed

        os.makedirs("recordings", exist_ok=True)
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Prefer FLAC to save disk; fallback to WAV for broader codec support
        flac_path = os.path.join("recordings", base + ".flac")
        wav_path = os.path.join("recordings", base + ".wav")
        wrote = False
        try:
            sf.write(
                flac_path,
                trimmed.astype(np.float32),
                self.sr,
                format="FLAC",
                subtype="PCM_16",
            )
            self.current_audio_path = flac_path
            wrote = True
        except Exception:
            try:
                sf.write(
                    wav_path,
                    trimmed.astype(np.float32),
                    self.sr,
                    format="WAV",
                    subtype="PCM_16",
                )
                self.current_audio_path = wav_path
                wrote = True
            except Exception:
                self.current_audio_path = None

        self.player = AudioPlayer(self.sr)
        self.player.set_data(trimmed)

        x_env, y_env = self._envelope(trimmed)
        self.wave_line.setData(x_env, y_env)

        self.playhead.setPos(0)
        self._update_time_axis(trimmed.size)

        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(True)

        # Create a DB entry immediately after stopping, with empty scores
        try:
            sess = db.add_session(
                self.db,
                self.current_script_name,
                self.current_script_text,
                self.current_audio_path or "",
                transcript=None,
                wer=None,
                clarity=None,
                score=None,
            )
            self.current_session_id = sess.id
            label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
            it = QListWidgetItem(label)
            it.setData(QtCore.Qt.UserRole, sess.id)
            # remove placeholder if present
            if self.history_list.count() == 1 and not isinstance(
                self.history_list.item(0).data(QtCore.Qt.UserRole), int
            ):
                self.history_list.takeItem(0)
            self.history_list.addItem(it)
            self.metrics_label.setText("Saved session; scores pending. Run Score when ready.")
        except Exception as e:
            # Fallback for legacy DBs with NOT NULL constraints: store placeholders
            try:
                # reset the transaction after the failed flush/commit
                try:
                    self.db.rollback()
                except Exception:
                    pass
                sess = db.add_session(
                    self.db,
                    self.current_script_name,
                    self.current_script_text,
                    self.current_audio_path,
                    transcript="",
                    wer=0.0,
                    clarity=0.0,
                    score=0.0,
                )
                self.current_session_id = sess.id
                label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
                it = QListWidgetItem(label)
                it.setData(QtCore.Qt.UserRole, sess.id)
                if self.history_list.count() == 1 and not isinstance(
                    self.history_list.item(0).data(QtCore.Qt.UserRole), int
                ):
                    self.history_list.takeItem(0)
                self.history_list.addItem(it)
                self.metrics_label.setText("Saved session (legacy DB); scores pending.")
            except Exception as e2:
                # Surface the specific error for troubleshooting
                self.metrics_label.setText(
                    f"Could not save session. DB error: {e2}"
                )
                try:
                    print("DB save error (first):", repr(e))
                    print("DB save error (fallback):", repr(e2))
                except Exception:
                    pass

    def _trim_silence(self, data, thresh=0.02):
        mask = np.abs(data) > thresh
        if not mask.any():
            return data
        i1 = np.argmax(mask)
        i2 = len(mask) - np.argmax(mask[::-1])
        return data[i1:i2]

    def _update_waveform(self) -> None:
        if not self.audio_buffer:
            return
        snippet = np.concatenate(self.audio_buffer, axis=0).flatten()
        snippet = snippet[-self.sr :]
        x = np.arange(snippet.size) / self.sr
        x_env, y_env = self._envelope(snippet)
        self.wave_line.setData(x_env, y_env)
        self.playhead.setPos(x_env[-1])
        self._update_time_axis(snippet.size)

    # --------------------------- playback ---------------------------------

    def _toggle_play_pause(self) -> None:
        if self.audio_data is None:
            return
        if self.player.active:
            self.player.pause()
            self.play_timer.stop()
            self.btn_play.setIcon(self.ic_play)
            return
        start = (
            self.player.idx
            if 0 <= self.player.idx < self.audio_data.size
            else 0
        )
        self.player.set_data(self.audio_data)
        self.player.play(start)
        self.play_timer.start()
        self.btn_play.setIcon(self.ic_pause)

    def _vb_click_event(self, ev) -> None:
        if (
            ev.button() != QtCore.Qt.LeftButton
            or self.audio_data is None
        ):
            ev.ignore()
            return
        pos = ev.scenePos()
        vb = self.plot.getPlotItem().getViewBox()
        t = vb.mapSceneToView(pos).x()
        t = max(0.0, min(t, self.audio_data.size / self.sr))
        idx = int(t * self.sr)

        self.playhead.setPos(t)
        self.player.set_data(self.audio_data)
        self.player.play(idx)
        self.play_timer.start()
        self.btn_play.setIcon(self.ic_pause)
        ev.accept()

    def _update_playhead(self) -> None:
        if not self.player.active:
            self.play_timer.stop()
            self.btn_play.setIcon(self.ic_play)
            # auto-rewind to start visually when finished
            try:
                if self.audio_data is not None and self.player.idx >= self.audio_data.size:
                    self.playhead.setPos(0)
            except Exception:
                pass
            return
        self.playhead.setPos(self.player.idx / self.sr)

    # --------------------------- scoring ----------------------------------

    def _transcribe_and_score(self) -> None:
        if self.current_audio_path is None:
            return
        self.btn_score.setEnabled(False)
        self.metrics_label.setText("Scoring… please wait")
        self.worker = TranscribeWorker(
            self.model,
            self.current_script_text,
            self.current_audio_path,
            self,
        )
        self.worker.completed.connect(self._on_transcription_done)
        self.worker.start()

    @QtCore.pyqtSlot(str, float, float, float)
    def _on_transcription_done(
        self, hyp: str, err: float, clar: float, score: float
    ) -> None:
        self.metrics_label.setText(
            f"Score: {score:.2f}/5 | WER: {err:.2%} | Clarity: {clar:.2%}"
        )
        self.transcript_txt.setText(hyp)
        self.btn_score.setEnabled(True)

        # Update existing session if it was created at stop; otherwise create
        if getattr(self, "current_session_id", None) is not None:
            sess = db.update_session_scores(
                self.db,
                self.current_session_id,
                hyp,
                err,
                clar,
                score,
            )
        else:
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
            self.current_session_id = sess.id
            label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
            it = QListWidgetItem(label)
            it.setData(QtCore.Qt.UserRole, sess.id)
            if self.history_list.count() == 1 and not isinstance(
                self.history_list.item(0).data(QtCore.Qt.UserRole), int
            ):
                self.history_list.takeItem(0)
            self.history_list.addItem(it)

    # ----------------------- axis helper ----------------------------------

    def _update_time_axis(self, n_samples: int) -> None:
        dur = n_samples / self.sr if n_samples else 15
        step = 1.0 if dur <= 60 else 5.0
        ticks = [(t, f"{int(t)}") for t in np.arange(0, dur + 0.1, step)]
        self.top_axis.setTicks([ticks, []])
        self.plot.setLimits(xMin=0, xMax=max(1, dur))

    def _envelope(
            self,
            y: np.ndarray,
            max_points: int = 4_000,
            silence_eps: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Down-sample y to ≤ max_points vertices while preserving peaks.
        Windows whose (max-min) < silence_eps are treated as silence
        and collapsed to a single y = 0 value so the plot stays flat.
        Returns (x, y) ready for setData().
        """
        n = y.size
        if n == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        if n <= max_points:
            x = np.arange(n) / self.sr
            return x, y

        step = int(np.ceil(n / max_points))  # samples / bucket
        win = y[: (n // step) * step].reshape(-1, step)
        y_min = win.min(axis=1)
        y_max = win.max(axis=1)
        dyn = y_max - y_min  # window range

        # flatten windows that are essentially silent
        quiet = dyn < silence_eps
        y_min[quiet] = 0.0
        y_max[quiet] = 0.0

        # interleave [min, max] so the poly goes min→max→min→max→…
        y_env = np.empty(y_min.size * 2, dtype=y.dtype)
        y_env[0::2] = y_min
        y_env[1::2] = y_max

        centres = (np.arange(y_min.size) * step + step // 2) / self.sr
        x_env = np.repeat(centres, 2)

        return x_env, y_env

    # ----------------------- load session ---------------------------------

    def _load_session(self, item) -> None:
        sid = item.data(QtCore.Qt.UserRole)
        sess = db.get_session_by_id(self.db, sid)
        self.current_session_id = sid

        if hasattr(self, "player"):
            self.player.pause()
        self.play_timer.stop()

        self.current_script_name = sess.script_name
        self.current_script_text = sess.script_text
        self.script_txt.setText(sess.script_text)
        # Handle sessions that may not yet have scores/transcript
        no_scores = (sess.transcript is None) or (sess.transcript == "")
        score_txt = (
            f"{sess.score:.2f}" if (sess.score is not None and not no_scores) else "–"
        )
        wer_txt = (
            f"{sess.wer:.2%}" if (sess.wer is not None and not no_scores) else "–"
        )
        clar_txt = (
            f"{sess.clarity:.2%}" if (sess.clarity is not None and not no_scores) else "–"
        )
        self.metrics_label.setText(
            f"Score: {score_txt}/5 | WER: {wer_txt} | Clarity: {clar_txt}"
        )
        self.transcript_txt.setText(sess.transcript or "")

        # Load audio (WAV/FLAC/OGG/MP3...) and convert to mono float32 at app samplerate
        data, file_sr = sf.read(sess.audio_path, dtype="float32")
        if hasattr(data, "ndim") and data.ndim > 1:
            data = data[:, 0]
        if file_sr != self.sr and data.size > 0:
            duration = data.size / float(file_sr)
            new_len = int(round(duration * self.sr))
            x_old = np.linspace(0.0, duration, num=data.size, endpoint=False)
            x_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)
        self.audio_data = data
        self.current_audio_path = sess.audio_path

        x_env, y_env = self._envelope(data)
        self.wave_line.setData(x_env, y_env)

        self.playhead.setPos(0)
        self._update_time_axis(data.size)

        self.player = AudioPlayer(self.sr)
        self.player.set_data(data)

        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(True)


# ────────────────────────────── main ///////////////////////////////////////


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    # Apply modern theme before creating the window
    try:
        apply_modern_theme(app)
    except Exception:
        pass
    win = SpeechPracticeApp()
    win.resize(1100, 640)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()