# speech_practice.py
# prettier-ignore
from __future__ import annotations

import os
import sys
from datetime import datetime
import subprocess
import soundfile as sf
from pydub import AudioSegment
import tempfile

import numpy as np
import sounddevice as sd

from jiwer import wer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMenu,
    QComboBox,
    QCheckBox,
    QFormLayout,
    QDialogButtonBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import db
from audio_player import AudioPlayer
from script_loader import pick_next_script
from transcribe_worker import TranscribeWorker, FreeTranscribeWorker
from transcription_service import TranscriptionService
from theme import apply_modern_theme
from icons import (
    make_play_icon,
    make_pause_icon,
    make_stop_icon,
    make_record_icon,
)
from audio_utils import trim_silence, envelope
from transcript_utils import (
    build_transcript_from_segments,
    highlight_transcript_at_time,
)
from settings_ui import (
    default_settings,
    settings_path,
    load_settings,
    save_settings,
    open_settings_dialog,
    whisper_options,
)
from free_speak import (
    on_toggle_free_mode,
    transcribe_free,
    save_free_speak_session,
)
from progress_tracker import open_progress_tracker
import json


# ───────────────────────────── theming ───────────────────────────────────

# ───────────────────────────── main window ────────────────────────────────


def load_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio file supporting various formats including M4A.
    Returns (audio_data, sample_rate) as float32 mono.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Try direct soundfile loading first (for WAV, FLAC, OGG, etc.)
    try:
        return sf.read(file_path, dtype="float32")
    except Exception as e:
        # If soundfile fails and it's a format that needs pydub conversion
        if file_ext in ['.m4a', '.aac', '.mp3', '.wma', '.mp4', '.mov']:
            try:
                # Load with pydub and convert to temporary WAV for soundfile
                audio_segment = AudioSegment.from_file(file_path)
                
                # Convert to mono and get raw audio data
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_wav_path = temp_file.name
                    audio_segment.export(temp_wav_path, format="wav")
                
                try:
                    # Load the temporary WAV file with soundfile
                    data, sr = sf.read(temp_wav_path, dtype="float32")
                    return data, sr
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_wav_path)
                    except:
                        pass
                        
            except Exception as pydub_error:
                raise Exception(f"Failed to load audio file {file_path}. Soundfile error: {e}. Pydub error: {pydub_error}")
        else:
            # Re-raise original soundfile error for other formats
            raise e


class SpeechPracticeApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speech Clarity Practice")

        # back-end
        self.db = db.get_session()
        self.sr = 16_000
        # Transcription service handles all transcription operations
        self.transcription_service = TranscriptionService(self)

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
        # live waveform tail window (seconds) to avoid repeated O(n) concatenations
        self._live_tail_seconds: int = 1

        # modes/state
        self.free_speak_mode: bool = False
        self.last_transcript_text: str = ""
        # transcript timestamp sync state
        self.transcript_segments: list[dict] | None = None
        self.transcript_segment_ranges: list[tuple[int, int, float, float]] = []  # (start_char, end_char, t0, t1)
        self.transcript_active_index: int = -1

        self._build_ui()
        # housekeeping: remove any recording files not referenced by the DB
        try:
            self._cleanup_orphan_recordings()
        except Exception:
            pass
        self._load_history()
        self.load_next_script()
        # Load persisted settings
        self._load_settings()

    # ───────────────────────────────── UI ─────────────────────────────────

    def _build_ui(self) -> None:
        mb = self.menuBar()

        act_next = mb.addAction("Next Script")
        act_next.triggered.connect(self.load_next_script)
        # Free Speak mode toggle directly on the menubar
        self.act_free_mode = QAction("Free Speak Mode", self, checkable=True)
        self.act_free_mode.setStatusTip("Transcribe without scoring or auto-saving")
        self.act_free_mode.toggled.connect(lambda checked: on_toggle_free_mode(self, checked))
        mb.addAction(self.act_free_mode)
        # Progress Tracker action
        act_progress = mb.addAction("Progress Tracker")
        act_progress.triggered.connect(self._open_progress_tracker)
        # Settings dialog action
        act_settings = mb.addAction("Settings")
        act_settings.triggered.connect(self._open_settings)

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
        self.ic_play = make_play_icon()
        self.ic_pause = make_pause_icon()
        self.ic_stop = make_stop_icon()
        self.ic_record = make_record_icon()

        # buttons
        self.btn_record = QPushButton(objectName="RecordBtn")
        self.btn_record.setIcon(self.ic_record)
        self.btn_play = QPushButton(objectName="CircleBtn")
        self.btn_play.setIcon(self.ic_play)
        self.btn_score = QPushButton("Score")
        self.btn_score.setObjectName("PrimaryButton")
        # Save button (for Free Speak optional saving)
        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("PrimaryButton")
        # volume slider
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # 0..120 maps to ~0%..200% with perceptual curve
        self.vol_slider.setRange(0, 120)
        self.vol_slider.setValue(90)
        self.vol_slider.setFixedWidth(120)
        self.vol_slider.setToolTip("Volume")

        icon_sz = 28  # <— pick any size you like (px)

        # remake glyphs at the chosen size
        self.ic_record = make_record_icon(icon_sz)
        self.ic_play = make_play_icon(icon_sz)
        self.ic_pause = make_pause_icon(icon_sz)
        self.ic_stop = make_stop_icon(icon_sz)

        # enlarge the icons that live on the buttons
        self.btn_play.setIconSize(QtCore.QSize(icon_sz, icon_sz))
        self.btn_record.setIconSize(QtCore.QSize(icon_sz, icon_sz))

        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_score.clicked.connect(self.transcription_service.transcribe_and_score)
        self.btn_save.clicked.connect(lambda: save_free_speak_session(self))
        self.vol_slider.valueChanged.connect(self._on_volume_changed)

        for b in (self.btn_record, self.btn_play, self.btn_score, self.btn_save):
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

    def _replace_player(self, new_player: AudioPlayer) -> None:
        try:
            if hasattr(self, "player") and self.player is not None:
                self.player.close()
        except Exception:
            pass
        self.player = new_player

    

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
        sid = item.data(QtCore.Qt.UserRole)
        act_open = QAction("Open In Explorer", self)
        act_open.triggered.connect(lambda: self._open_in_explorer(sid))
        menu.addAction(act_open)
        act_del = QAction("Delete Session", self)
        act_del.triggered.connect(lambda: self._delete_session(item))
        menu.addAction(act_del)
        menu.exec_(self.history_list.mapToGlobal(pt))

    def _open_in_explorer(self, sess_id: int) -> None:
        try:
            sess = db.get_session_by_id(self.db, sess_id)
            if not sess:
                return
            path = os.path.abspath(sess.audio_path)
            if not os.path.exists(path):
                self.metrics_label.setText(f"File not found: {path}")
                return
            if sys.platform.startswith("win"):
                # Use Explorer to select the file
                subprocess.run(["explorer", "/select,", path.replace("/", "\\")])
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", path])
            else:
                subprocess.run(["xdg-open", os.path.dirname(path)])
        except Exception as e:
            self.metrics_label.setText(f"Could not open in Explorer: {e}")

    # --------------------------- housekeeping ------------------------------
    def _cleanup_orphan_recordings(self) -> None:
        rec_dir = "recordings"
        if not os.path.isdir(rec_dir):
            return
        # Collect referenced absolute paths from DB
        try:
            sessions = db.get_all_sessions(self.db)
        except Exception:
            sessions = []
        referenced: set[str] = set()
        for sess in sessions or []:
            try:
                if getattr(sess, "audio_path", None):
                    referenced.add(os.path.abspath(sess.audio_path))
            except Exception:
                continue
        # Delete files in recordings/ not in referenced
        deleted = 0
        for fname in os.listdir(rec_dir):
            fpath = os.path.abspath(os.path.join(rec_dir, fname))
            # Only consider typical audio files
            if not fname.lower().endswith((".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac", ".wma")):
                continue
            if fpath not in referenced:
                try:
                    os.remove(fpath)
                    deleted += 1
                except Exception:
                    pass
        if deleted:
            try:
                print(f"Cleaned {deleted} orphan recording(s) from '{rec_dir}'.")
            except Exception:
                pass

    # --------------------------- data ------------------------------------

    def _load_history(self) -> None:
        self.history_list.clear()
        sessions = db.get_all_sessions(self.db)
        if not sessions:
            self.history_list.addItem("No sessions yet")
        else:
            for sess in sessions:
                formatted_timestamp = datetime.strptime(sess.timestamp, "%Y-%m-%dT%H:%M:%S").strftime("%d %b %Y %H:%M")
                label = f"{formatted_timestamp} — {sess.script_name}"
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
        try:
            self.transcript_txt.setExtraSelections([])
        except Exception:
            pass
        self.transcript_segments = None
        self.transcript_segment_ranges = []
        self.transcript_active_index = -1
        self.audio_data = None
        self.current_audio_path = None

        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.btn_score.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_save.setVisible(self.free_speak_mode)

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
            trimmed = trim_silence(raw, self.sr)
            if trimmed.size < self.sr // 10:
                trimmed = raw
        self.audio_data = trimmed

        # In Free Speak mode, do not write to disk unless the user chooses Save later
        if not self.free_speak_mode:
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
        else:
            self.current_audio_path = None

        self._replace_player(AudioPlayer(self.sr))
        self.player.set_data(trimmed)

        x_env, y_env = envelope(trimmed, self.sr)
        self.wave_line.setData(x_env, y_env)

        self.playhead.setPos(0)
        self._update_time_axis(trimmed.size)

        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(True)
        if self.free_speak_mode:
            # In Free Speak, do not auto-save; auto-transcribe instead
            self.btn_score.setText("Transcribe")
            self.btn_save.setVisible(True)
            self.btn_save.setEnabled(False)
            self.metrics_label.setText("Free Speak: ready to transcribe")
            # Auto-start transcription for convenience
            QtCore.QTimer.singleShot(100, self.transcription_service.transcribe_free)
            return

        if not self.free_speak_mode:
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
    

    def _update_waveform(self) -> None:
        if not self.audio_buffer:
            return
        # Concatenate only the trailing blocks required for the last N seconds
        needed = int(self.sr * self._live_tail_seconds)
        total = 0
        parts: list[np.ndarray] = []
        for block in reversed(self.audio_buffer):
            parts.append(block)
            total += int(block.shape[0])
            if total >= needed:
                break
        snippet = (
            np.concatenate(list(reversed(parts)), axis=0).flatten()
            if parts
            else np.zeros(0, dtype=np.float32)
        )
        snippet = snippet[-needed:]
        x_env, y_env = envelope(snippet, self.sr)
        self.wave_line.setData(x_env, y_env)
        if x_env.size:
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
        cur_t = self.player.idx / self.sr
        self.playhead.setPos(cur_t)
        # update transcript highlighting if we have timestamped segments
        try:
            if self.transcript_segment_ranges:
                try:
                    self.transcript_active_index = highlight_transcript_at_time(
                        self.transcript_txt,
                        self.transcript_segment_ranges,
                        cur_t,
                        self.transcript_active_index,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # --------------------------- scoring ----------------------------------

    def _save_free_speak_session(self) -> None:
        save_free_speak_session(self)

    def _on_toggle_free_mode(self, checked: bool) -> None:
        on_toggle_free_mode(self, checked)

    # ----------------------- axis helper ----------------------------------

    def _update_time_axis(self, n_samples: int) -> None:
        dur = n_samples / self.sr if n_samples else 15
        step = 1.0 if dur <= 60 else 5.0
        ticks = [(t, f"{int(t)}") for t in np.arange(0, dur + 0.1, step)]
        self.top_axis.setTicks([ticks, []])
        self.plot.setLimits(xMin=0, xMax=max(1, dur))

    

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
        try:
            self.transcript_txt.setExtraSelections([])
        except Exception:
            pass
        # Restore segments if present in DB to enable synced highlighting
        try:
            if getattr(sess, "segments", None):
                segs_from_db = json.loads(sess.segments)
                seg_list = list(segs_from_db) if isinstance(segs_from_db, list) else []
                txt, segs, ranges, active_idx = build_transcript_from_segments(seg_list)
                self.transcript_segments = segs
                self.transcript_segment_ranges = ranges
                self.transcript_active_index = -1
                if not sess.transcript:
                    self.transcript_txt.setPlainText(txt)
        except Exception:
            self.transcript_segments = None
            self.transcript_segment_ranges = []
        self.transcript_active_index = -1

        # Load audio (all formats: WAV/FLAC/OGG/AIFF via soundfile, MP3/M4A/AAC/WMA via pydub) and convert to mono float32 at app samplerate
        data, file_sr = load_audio_file(sess.audio_path)
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

        x_env, y_env = envelope(data, self.sr)
        self.wave_line.setData(x_env, y_env)

        self.playhead.setPos(0)
        self._update_time_axis(data.size)

        self._replace_player(AudioPlayer(self.sr))
        self.player.set_data(data)

        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_score.setEnabled(True)

    # --------------------------- settings ----------------------------------

    def _default_settings(self) -> dict:
        return default_settings()

    def _settings_path(self) -> str:
        return settings_path()

    def _load_settings(self) -> None:
        self.settings = load_settings(self._default_settings(), self._settings_path())

    def _save_settings(self) -> None:
        save_settings(self.settings, self._settings_path())

    def _detect_gpu(self) -> tuple[bool, str]:
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                total_vram = int(torch.cuda.get_device_properties(0).total_memory // (1024 ** 2))
                return True, f"{name} ({total_vram} MB VRAM, {count} device(s))"
        except Exception:
            pass
        return False, "No CUDA GPU detected"

    def _open_progress_tracker(self) -> None:
        """Open the progress tracker dialog."""
        open_progress_tracker(self)
    
    def _open_settings(self) -> None:
        open_settings_dialog(self)




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
    ret = app.exec_()
    # best-effort resource cleanup
    try:
        if hasattr(win, "player") and win.player is not None:
            win.player.close()
        if hasattr(win, "stream") and win.stream is not None:
            try:
                win.stream.stop(); win.stream.close()
            except Exception:
                pass
        try:
            win.db.close()
        except Exception:
            pass
    except Exception:
        pass
    sys.exit(ret)


if __name__ == "__main__":
    main()