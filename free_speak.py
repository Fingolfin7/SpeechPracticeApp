from __future__ import annotations

import json
from datetime import datetime
import os
from typing import Optional

import numpy as np
import soundfile as sf
from PyQt5 import QtCore
from PyQt5.QtWidgets import QListWidgetItem

import db


def on_toggle_free_mode(window, checked: bool) -> None:
    window.free_speak_mode = bool(checked)
    if window.free_speak_mode:
        try:
            window.act_free_mode.setText("Script Mode")
            window.act_free_mode.setStatusTip("Switch back to scripted practice")
        except Exception:
            pass
        window.btn_score.setText("Transcribe")
        window.btn_save.setVisible(True)
        window.btn_save.setEnabled(False)
        window.metrics_label.setText("Free Speak: not scoring")
        window.script_txt.setText("Free Speak Mode — Talk, then Transcribe.")
        window.current_session_id = None
    else:
        try:
            window.act_free_mode.setText("Free Speak Mode")
            window.act_free_mode.setStatusTip("Transcribe without scoring or auto-saving")
        except Exception:
            pass
        window.btn_score.setText("Score")
        window.btn_save.setVisible(False)
        window.load_next_script()


def transcribe_free(window) -> None:
    # Delegate to the transcription service
    window.transcription_service.transcribe_free()


def save_free_speak_session(window) -> None:
    if not window.free_speak_mode:
        return
    transcript = window.transcript_txt.toPlainText().strip()
    if window.current_audio_path is None:
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("recordings", exist_ok=True)
        try:
            flac_path = os.path.join("recordings", base + ".flac")
            sf.write(
                flac_path,
                (window.audio_data or np.zeros(0, dtype=np.float32)).astype(np.float32),
                window.sr,
                format="FLAC",
                subtype="PCM_16",
            )
            window.current_audio_path = flac_path
        except Exception:
            try:
                wav_path = os.path.join("recordings", base + ".wav")
                sf.write(
                    wav_path,
                    (window.audio_data or np.zeros(0, dtype=np.float32)).astype(np.float32),
                    window.sr,
                    format="WAV",
                    subtype="PCM_16",
                )
                window.current_audio_path = wav_path
            except Exception as e:
                window.metrics_label.setText(f"Could not save recording file: {e}")
                return
    try:
        segments_json: Optional[str] = None
        if window.transcript_segments is not None and window.transcript_segment_ranges:
            try:
                segments_json = json.dumps(window.transcript_segments)
            except Exception:
                segments_json = None
        sess = db.add_session(
            window.db,
            "Free Speak",
            "",
            window.current_audio_path,
            transcript=transcript if transcript else None,
            wer=None,
            clarity=None,
            score=None,
            segments_json=segments_json,
        )
        window.current_session_id = sess.id
        label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
        it = QListWidgetItem(label)
        it.setData(QtCore.Qt.UserRole, sess.id)
        if window.history_list.count() == 1 and not isinstance(
            window.history_list.item(0).data(QtCore.Qt.UserRole), int
        ):
            window.history_list.takeItem(0)
        window.history_list.addItem(it)
        window.metrics_label.setText("Saved Free Speak session")
    except Exception as e:
        window.metrics_label.setText(f"Could not save Free Speak session: {e}")


