from __future__ import annotations

import json
import os
from typing import Optional, Callable, Any, Dict

import whisper
from PyQt5 import QtCore
from PyQt5.QtWidgets import QListWidgetItem

import db
from transcribe_worker import TranscribeWorker, FreeTranscribeWorker
from transcript_utils import build_transcript_from_segments, highlight_transcript_at_time
from settings_ui import whisper_options


class TranscriptionService:
    """
    Centralized service for handling all transcription operations.
    This service manages Whisper model loading, transcription workflows,
    and result processing for both script-based and free-speak modes.
    """
    
    def __init__(self, window):
        """Initialize the transcription service with a reference to the main window."""
        self.window = window
        self.model: Optional[Any] = None
        self._received_segments_this_run: bool = False
    
    def ensure_model(self) -> None:
        """Lazy-load Whisper model to reduce startup time and memory usage."""
        if self.model is None:
            model_name = (
                getattr(self.window, "settings", {}).get("model_name")
                if hasattr(self.window, "settings") and isinstance(self.window.settings, dict)
                else None
            ) or os.getenv("WHISPER_MODEL", "base.en")
            self.model = whisper.load_model(model_name)
    
    def get_whisper_options(self, free_speak: bool = False) -> dict:
        """Get Whisper transcription options from settings."""
        return whisper_options(self.window.settings, free_speak=free_speak)
    
    def transcribe_and_score(self) -> None:
        """
        Main transcription workflow for script-based practice.
        Transcribes audio and computes scores against reference text.
        """
        if self.window.current_audio_path is None:
            return
        
        if self.window.free_speak_mode:
            self.transcribe_free()
            return
        
        self.window.btn_score.setEnabled(False)
        self.window.metrics_label.setText("Scoring… please wait")
        self.ensure_model()
        
        self.window.worker = TranscribeWorker(
            self.model,
            self.window.current_script_text,
            self.window.current_audio_path,
            self.window,
            options=self.get_whisper_options(),
        )
        self._received_segments_this_run = False
        
        try:
            self.window.worker.completed_with_segments.connect(self.on_transcription_done_with_segments)
        except Exception:
            pass
        
        self.window.worker.completed.connect(self.on_transcription_done)
        self.window.worker.start()
    
    def transcribe_free(self) -> None:
        """
        Transcription workflow for free-speak mode.
        Transcribes audio without scoring against reference text.
        """
        if self.window.audio_data is None and self.window.current_audio_path is None:
            return
        
        self.window.btn_score.setEnabled(False)
        self.window.metrics_label.setText("Transcribing… (Free Speak)")
        
        audio_input = (
            self.window.audio_data 
            if self.window.audio_data is not None 
            else self.window.current_audio_path
        )
        
        self.ensure_model()
        
        self.window.free_worker = FreeTranscribeWorker(
            self.model,
            audio_input,
            self.window,
            options=self.get_whisper_options(free_speak=True),
        )
        self._received_segments_this_run = False
        
        try:
            self.window.free_worker.completed_with_segments.connect(
                self.on_free_transcription_done_with_segments
            )
        except Exception:
            pass
        
        self.window.free_worker.completed.connect(self.on_free_transcription_done)
        self.window.free_worker.start()
    
    @QtCore.pyqtSlot(str, float, float, float)
    def on_transcription_done(
        self, hyp: str, err: float, clar: float, score: float
    ) -> None:
        """Handle completion of script-based transcription with scores."""
        self.window.metrics_label.setText(
            f"Score: {score:.2f}/5 | WER: {err:.2%} | Clarity: {clar:.2%}"
        )
        
        if not self._received_segments_this_run:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        
        self.window.btn_score.setEnabled(True)
        
        # Update existing session if it was created at stop; otherwise create
        if getattr(self.window, "current_session_id", None) is not None:
            # Persist segments if we used them in this run
            segments_json = None
            if (self.window.transcript_segments is not None and 
                self.window.transcript_segment_ranges):
                try:
                    segments_json = json.dumps(self.window.transcript_segments)
                except Exception:
                    segments_json = None
            
            sess = db.update_session_scores(
                self.window.db,
                self.window.current_session_id,
                hyp,
                err,
                clar,
                score,
                segments_json=segments_json,
            )
        else:
            segments_json = None
            if (self.window.transcript_segments is not None and 
                self.window.transcript_segment_ranges):
                try:
                    segments_json = json.dumps(self.window.transcript_segments)
                except Exception:
                    segments_json = None
            
            sess = db.add_session(
                self.window.db,
                self.window.current_script_name,
                self.window.current_script_text,
                self.window.current_audio_path,
                hyp,
                err,
                clar,
                score,
                segments_json=segments_json,
            )
            self.window.current_session_id = sess.id
            label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
            it = QListWidgetItem(label)
            it.setData(QtCore.Qt.UserRole, sess.id)
            
            if (self.window.history_list.count() == 1 and 
                not isinstance(self.window.history_list.item(0).data(QtCore.Qt.UserRole), int)):
                self.window.history_list.takeItem(0)
            
            self.window.history_list.addItem(it)
    
    @QtCore.pyqtSlot(str)
    def on_free_transcription_done(self, hyp: str) -> None:
        """Handle completion of free-speak transcription."""
        self.window.last_transcript_text = hyp
        
        if not self._received_segments_this_run:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        
        self.window.metrics_label.setText("Transcript ready (Free Speak)")
        self.window.btn_score.setEnabled(True)
        self.window.btn_save.setEnabled(True)
    
    @QtCore.pyqtSlot(str, float, float, float, object)
    def on_transcription_done_with_segments(
        self, hyp: str, err: float, clar: float, score: float, segments: object
    ) -> None:
        """Handle completion of script-based transcription with word-level segments."""
        try:
            self._received_segments_this_run = True
            txt, segs, ranges, active_idx = build_transcript_from_segments(segments)
            self.window.transcript_segments = segs
            self.window.transcript_segment_ranges = ranges
            self.window.transcript_active_index = active_idx
            self.window.transcript_txt.setPlainText(txt)
            
            self.window.metrics_label.setText(
                f"Score: {score:.2f}/5 | WER: {err:.2%} | Clarity: {clar:.2%}"
            )
            
            self.window.transcript_active_index = -1
            self.window.transcript_active_index = highlight_transcript_at_time(
                self.window.transcript_txt, 
                self.window.transcript_segment_ranges, 
                0.0, 
                self.window.transcript_active_index
            )
            
            # Persist on-the-fly if we already have a session id
            if getattr(self.window, "current_session_id", None) is not None:
                try:
                    segments_json = json.dumps(self.window.transcript_segments or [])
                    db.update_session_scores(
                        self.window.db,
                        self.window.current_session_id,
                        hyp,
                        err,
                        clar,
                        score,
                        segments_json=segments_json,
                    )
                except Exception:
                    pass
        except Exception:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        finally:
            self.window.btn_score.setEnabled(True)
    
    @QtCore.pyqtSlot(str, object)
    def on_free_transcription_done_with_segments(self, hyp: str, segments: object) -> None:
        """Handle completion of free-speak transcription with word-level segments."""
        try:
            self._received_segments_this_run = True
            self.window.last_transcript_text = hyp
            txt, segs, ranges, active_idx = build_transcript_from_segments(segments)
            self.window.transcript_segments = segs
            self.window.transcript_segment_ranges = ranges
            self.window.transcript_active_index = active_idx
            self.window.transcript_txt.setPlainText(txt)
            self.window.metrics_label.setText("Transcript ready (Free Speak)")
            
            self.window.transcript_active_index = -1
            self.window.transcript_active_index = highlight_transcript_at_time(
                self.window.transcript_txt, 
                self.window.transcript_segment_ranges, 
                0.0, 
                self.window.transcript_active_index
            )
            # In free speak, we don't have a session yet; persistence happens on Save
        except Exception:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        finally:
            self.window.btn_score.setEnabled(True)
            self.window.btn_save.setEnabled(True)
    
    def _clear_transcript_sync(self) -> None:
        """Clear transcript synchronization state."""
        self.window.transcript_segments = None
        self.window.transcript_segment_ranges = []
        self.window.transcript_active_index = -1
        try:
            self.window.transcript_txt.setExtraSelections([])
        except Exception:
            pass
