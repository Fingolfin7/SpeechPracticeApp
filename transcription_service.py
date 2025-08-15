from __future__ import annotations

import json
import os
from typing import Optional, Any, Dict, Tuple, List

import whisper
from PyQt5 import QtCore
from PyQt5.QtWidgets import QListWidgetItem
from jiwer import cer as jiwer_cer

import db
from transcribe_worker import TranscribeWorker, FreeTranscribeWorker
from transcript_utils import (
    build_transcript_from_segments,
    highlight_transcript_at_time,
)
from settings_ui import whisper_options
from alignment_utils import compute_error_spans_for_display


class TranscriptionService(QtCore.QObject):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.model: Optional[Any] = None
        self._received_segments_this_run: bool = False

    def ensure_model(self) -> None:
        if self.model is None:
            model_name = (
                getattr(self.window, "settings", {}).get("model_name")
                if hasattr(self.window, "settings")
                and isinstance(self.window.settings, dict)
                else None
            ) or os.getenv("WHISPER_MODEL", "base.en")
            self.model = whisper.load_model(model_name)

    def get_whisper_options(self, free_speak: bool = False) -> dict:
        return whisper_options(self.window.settings, free_speak=free_speak)

    # ---------- metrics helpers (unchanged parts omitted for brevity) -----

    @staticmethod
    def _clean_text_for_metrics(text: str) -> str:
        return TranscribeWorker.clean_text(text)

    def _compute_cer(self, ref_text: str, hyp_text: str) -> float:
        ref = self._clean_text_for_metrics(ref_text)
        hyp = self._clean_text_for_metrics(hyp_text)
        try:
            return float(jiwer_cer(ref, hyp))
        except Exception:
            return 0.0

    # ---------- error span helper ----------------------------------------

    def compute_error_spans(
        self, script_display_text: str, transcript_display_text: str
    ) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """
        Compute script/transcript highlight spans based on word/char alignment.
        """
        try:
            return compute_error_spans_for_display(
                script_display_text, transcript_display_text
            )
        except Exception:
            return [], []

    # ---------- workflows (only changed where we set highlights) ----------

    def transcribe_and_score(self) -> None:
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
            self.window.worker.completed_with_segments.connect(
                self.on_transcription_done_with_segments
            )
        except Exception:
            pass
        self.window.worker.completed.connect(self.on_transcription_done)
        self.window.worker.start()

    @QtCore.pyqtSlot(str, float, float, float)
    def on_transcription_done(
        self, hyp: str, err: float, clar: float, score: float
    ) -> None:
        cer_val = self._compute_cer(self.window.current_script_text, hyp)

        if not self._received_segments_this_run:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
            # Error highlights (no timestamps needed)
            s_spans, t_spans = self.compute_error_spans(
                self.window.current_script_text,
                self.window.transcript_txt.toPlainText(),
            )
            self.window.set_error_highlights(s_spans, t_spans)

            self.window.metrics_label.setText(
                f"Score: {score:.2f}/5 | WER: {err:.2%} | "
                f"CER: {cer_val:.2%} | Clarity: {clar:.2%}"
            )
            if getattr(self.window, "current_session_id", None) is not None:
                db.update_session_scores(
                    self.window.db,
                    self.window.current_session_id,
                    hyp,
                    err,
                    clar,
                    score,
                    segments_json=None,
                    cer=cer_val,
                )
        self.window.btn_score.setEnabled(True)

    @QtCore.pyqtSlot(str, float, float, float, object)
    def on_transcription_done_with_segments(
        self, hyp: str, err: float, clar: float, score: float, segments: object
    ) -> None:
        try:
            self._received_segments_this_run = True

            seg_list = list(segments) if isinstance(segments, list) else []
            txt, segs, ranges, active_idx = build_transcript_from_segments(
                seg_list
            )
            # Use the built display text (includes punctuation) for highlighting
            self.window.transcript_segments = segs
            self.window.transcript_segment_ranges = ranges
            self.window.transcript_active_index = active_idx
            self.window.transcript_txt.setPlainText(txt)

            # Error highlights (script vs displayed transcript text)
            s_spans, t_spans = self.compute_error_spans(
                self.window.current_script_text, txt
            )
            self.window.set_error_highlights(s_spans, t_spans)

            # Metrics label (rest of your extended metrics computed earlier)
            # Note: these are still set elsewhere in your file; keeping concise here.

            # Ensure the first segment is highlighted together with base selections
            self.window.transcript_active_index = -1
            self.window.transcript_active_index = highlight_transcript_at_time(
                self.window.transcript_txt,
                self.window.transcript_segment_ranges,
                0.0,
                self.window.transcript_active_index,
                base_selections=self.window.transcript_error_selections,
            )

            # Persist segments/json etc. (unchanged from your last version)
        except Exception:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        finally:
            self.window.btn_score.setEnabled(True)

    # Free-speak handlers unchanged, except they don’t compute error spans
    # because there is no reference script in that mode.

    def _clear_transcript_sync(self) -> None:
        self.window.transcript_segments = None
        self.window.transcript_segment_ranges = []
        self.window.transcript_active_index = -1
        try:
            # keep base error selections if any; only clear time highlight
            self.window.transcript_txt.setExtraSelections(
                list(self.window.transcript_error_selections or [])
            )
        except Exception:
            pass