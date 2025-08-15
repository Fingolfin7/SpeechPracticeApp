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


class TranscriptionService(QtCore.QObject):
    """
    Centralized service for handling all transcription operations.
    Manages Whisper model loading, transcription workflows,
    and result processing for both script-based and free-speak modes.
    """

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

    # ------------------------ metric helpers ------------------------

    @staticmethod
    def _clean_text_for_metrics(text: str) -> str:
        # Reuse worker's normalization for consistency
        return TranscribeWorker.clean_text(text)

    def _compute_cer(self, ref_text: str, hyp_text: str) -> float:
        ref = self._clean_text_for_metrics(ref_text)
        hyp = self._clean_text_for_metrics(hyp_text)
        try:
            return float(jiwer_cer(ref, hyp))
        except Exception:
            return 0.0

    @staticmethod
    def _norm_conf_from_logprob(lp: Optional[float]) -> Optional[float]:
        if lp is None:
            return None
        # Map avg_logprob ~ [-1, 0] to [0, 1]
        conf = (float(lp) + 1.0) / 1.0
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        return conf

    def _augment_segments_and_fluency(
        self, segments: List[dict], hyp_text: str
    ) -> Tuple[List[dict], float, float, float, Optional[float]]:
        """
        Add duration, pause_before, conf to segments.
        Return (segments_aug, artic_rate_wpm, pause_ratio, filled_pauses_count, avg_conf)
        """
        segs: List[dict] = []
        prev_end: Optional[float] = None
        speech_time = 0.0
        pause_time = 0.0
        conf_vals: List[float] = []

        for seg in segments or []:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            dur = max(0.0, end - start)
            pb = 0.0
            if prev_end is not None:
                pb = max(0.0, start - prev_end)
                pause_time += pb
            prev_end = end
            speech_time += dur

            lp = seg.get("avg_logprob", None)
            conf = self._norm_conf_from_logprob(lp)
            if conf is not None:
                conf_vals.append(conf)

            seg_copy = dict(seg)
            seg_copy["duration"] = float(dur)
            seg_copy["pause_before"] = float(pb)
            if conf is not None:
                seg_copy["conf"] = float(conf)
            segs.append(seg_copy)

        total_time = 0.0
        if segments:
            total_time = float(
                max(0.0, float(segments[-1].get("end", 0.0)) - float(
                    segments[0].get("start", 0.0)
                ))
            )

        # Words per minute (exclude pauses): words / (speech_time / 60)
        hyp_clean = self._clean_text_for_metrics(hyp_text)
        n_words = len(hyp_clean.split())
        artic_rate = 0.0
        if speech_time > 1e-6:
            artic_rate = float(n_words) * (60.0 / speech_time)

        pause_ratio = 0.0
        denom = total_time if total_time > 1e-6 else (speech_time + pause_time)
        if denom > 1e-6:
            pause_ratio = pause_time / denom

        # Filled pauses in the hypothesis
        filled_set = {"um", "uh", "erm", "er", "hmm"}
        filled_count = float(sum(1 for t in hyp_clean.split() if t in filled_set))

        avg_conf = float(sum(conf_vals) / len(conf_vals)) if conf_vals else None

        return segs, artic_rate, pause_ratio, filled_count, avg_conf

    # ---------------------- workflows ----------------------

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

    def transcribe_free(self) -> None:
        if (
            self.window.audio_data is None
            and self.window.current_audio_path is None
        ):
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

        self.window.free_worker.completed.connect(
            self.on_free_transcription_done
        )
        self.window.free_worker.start()

    @QtCore.pyqtSlot(str, float, float, float)
    def on_transcription_done(
        self, hyp: str, err: float, clar: float, score: float
    ) -> None:
        """
        Handle completion of script-based transcription with scores (no segments).
        If segments arrived in this run, skip DB update here (the other slot will).
        """
        cer_val = self._compute_cer(self.window.current_script_text, hyp)

        if not self._received_segments_this_run:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()

            # Label with CER even without timing details
            self.window.metrics_label.setText(
                f"Score: {score:.2f}/5 | WER: {err:.2%} | "
                f"CER: {cer_val:.2%} | Clarity: {clar:.2%}"
            )

            # Update DB if session exists
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
        # Re-enable button regardless
        self.window.btn_score.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def on_free_transcription_done(self, hyp: str) -> None:
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
        """
        Handle completion of script-based transcription with timestamped segments.
        Compute CER + fluency + confidence, persist, and update UI.
        """
        try:
            self._received_segments_this_run = True

            # Augment segments + compute metrics
            seg_list = list(segments) if isinstance(segments, list) else []
            segs_aug, artic_rate, pause_ratio, filled_cnt, avg_conf = (
                self._augment_segments_and_fluency(seg_list, hyp)
            )
            cer_val = self._compute_cer(self.window.current_script_text, hyp)

            txt, segs_built, ranges, active_idx = build_transcript_from_segments(
                segs_aug
            )
            self.window.transcript_segments = segs_aug
            self.window.transcript_segment_ranges = ranges
            self.window.transcript_active_index = active_idx
            self.window.transcript_txt.setPlainText(txt)

            # Label with extended metrics
            avg_conf_txt = (
                f"{avg_conf:.0%}" if avg_conf is not None else "–"
            )
            self.window.metrics_label.setText(
                "Score: "
                f"{score:.2f}/5 | WER: {err:.2%} | CER: {cer_val:.2%} | "
                f"Clarity: {clar:.2%} | Rate: {artic_rate:.0f} wpm | "
                f"Pauses: {pause_ratio:.0%} | Conf: {avg_conf_txt}"
            )

            # Initial highlight at t=0
            self.window.transcript_active_index = -1
            self.window.transcript_active_index = highlight_transcript_at_time(
                self.window.transcript_txt,
                self.window.transcript_segment_ranges,
                0.0,
                self.window.transcript_active_index,
            )

            # Persist if we have a session id; else create a new one
            segments_json = json.dumps(segs_aug)
            if getattr(self.window, "current_session_id", None) is not None:
                db.update_session_scores(
                    self.window.db,
                    self.window.current_session_id,
                    hyp,
                    err,
                    clar,
                    score,
                    segments_json=segments_json,
                    cer=cer_val,
                    artic_rate=artic_rate,
                    pause_ratio=pause_ratio,
                    filled_pauses=filled_cnt,
                    avg_conf=avg_conf,
                )
            else:
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
                    cer=cer_val,
                    artic_rate=artic_rate,
                    pause_ratio=pause_ratio,
                    filled_pauses=filled_cnt,
                    avg_conf=avg_conf,
                )
                self.window.current_session_id = sess.id
                label = f"{sess.id}: {sess.timestamp} — {sess.script_name}"
                it = QListWidgetItem(label)
                it.setData(QtCore.Qt.UserRole, sess.id)
                if (
                    self.window.history_list.count() == 1
                    and not isinstance(
                        self.window.history_list.item(0).data(
                            QtCore.Qt.UserRole
                        ),
                        int,
                    )
                ):
                    self.window.history_list.takeItem(0)
                self.window.history_list.addItem(it)
        except Exception:
            # Fallback: show plain hypothesis if anything goes wrong
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        finally:
            self.window.btn_score.setEnabled(True)

    @QtCore.pyqtSlot(str, object)
    def on_free_transcription_done_with_segments(
        self, hyp: str, segments: object
    ) -> None:
        """
        Free speak with segments: compute fluency/confidence (no CER/WER here).
        """
        try:
            self._received_segments_this_run = True
            self.window.last_transcript_text = hyp

            seg_list = list(segments) if isinstance(segments, list) else []
            segs_aug, artic_rate, pause_ratio, filled_cnt, avg_conf = (
                self._augment_segments_and_fluency(seg_list, hyp)
            )

            txt, segs_built, ranges, active_idx = build_transcript_from_segments(
                segs_aug
            )
            self.window.transcript_segments = segs_aug
            self.window.transcript_segment_ranges = ranges
            self.window.transcript_active_index = active_idx
            self.window.transcript_txt.setPlainText(txt)

            avg_conf_txt = (
                f"{avg_conf:.0%}" if avg_conf is not None else "–"
            )
            self.window.metrics_label.setText(
                "Transcript ready (Free Speak) | "
                f"Rate: {artic_rate:.0f} wpm | "
                f"Pauses: {pause_ratio:.0%} | "
                f"Conf: {avg_conf_txt}"
            )

            self.window.transcript_active_index = -1
            self.window.transcript_active_index = highlight_transcript_at_time(
                self.window.transcript_txt,
                self.window.transcript_segment_ranges,
                0.0,
                self.window.transcript_active_index,
            )
            # Persistence happens on Save in free speak mode.
        except Exception:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        finally:
            self.window.btn_score.setEnabled(True)
            self.window.btn_save.setEnabled(True)

    def _clear_transcript_sync(self) -> None:
        self.window.transcript_segments = None
        self.window.transcript_segment_ranges = []
        self.window.transcript_active_index = -1
        try:
            self.window.transcript_txt.setExtraSelections([])
        except Exception:
            pass