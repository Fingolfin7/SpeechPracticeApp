from __future__ import annotations

import json
import os
import time
import gc
from datetime import datetime
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
from error_analytics import extract_error_events


class TranscriptionService(QtCore.QObject):
    """
    Centralized service for handling transcription and scoring.
    Adds extended metrics (CER, fluency, confidence) and error highlights.
    """

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.model: Optional[Any] = None
        self._received_segments_this_run: bool = False

    # ------------------------ model/options ------------------------

    def _resolve_model_device(self) -> str:
        """
        Resolve Whisper model device from settings:
        - gpu  -> cuda (fallback cpu if unavailable)
        - cpu  -> cpu
        - auto -> cuda if available else cpu
        """
        pref = (
            getattr(self.window, "settings", {}).get("device", "auto")
            if hasattr(self.window, "settings")
            and isinstance(self.window.settings, dict)
            else "auto"
        )
        if pref == "cpu":
            return "cpu"
        try:
            import torch

            has_cuda = bool(torch.cuda.is_available())
        except Exception:
            has_cuda = False
        if pref == "gpu":
            return "cuda" if has_cuda else "cpu"
        return "cuda" if has_cuda else "cpu"

    def ensure_model(self) -> None:
        if self.model is None:
            self._release_finished_workers()
            gc.collect()
            self._clear_cuda_cache()
            model_name = (
                getattr(self.window, "settings", {}).get("model_name")
                if hasattr(self.window, "settings")
                and isinstance(self.window.settings, dict)
                else None
            ) or os.getenv("WHISPER_MODEL", "base.en")
            device = self._resolve_model_device()
            try:
                self.model = whisper.load_model(model_name, device=device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._clear_cuda_cache()
                    gc.collect()
                    raise RuntimeError(
                        "CUDA out of memory while loading Whisper model. "
                        "Try a smaller model or wait for current transcriptions "
                        "to finish before switching models."
                    ) from e
                raise
            try:
                self.model.eval()
            except Exception:
                pass

    def get_whisper_options(self, free_speak: bool = False) -> dict:
        return whisper_options(self.window.settings, free_speak=free_speak)

    def _get_worker_timing(self, free_speak: bool = False) -> Dict[str, float]:
        worker_attr = "free_worker" if free_speak else "worker"
        worker = getattr(self.window, worker_attr, None)
        timing = getattr(worker, "last_timing", None)
        return dict(timing) if isinstance(timing, dict) else {}

    @staticmethod
    def _format_timing_text(timing: Dict[str, float], ui_post_s: float) -> str:
        asr_s = float(timing.get("asr_s", 0.0))
        worker_post_s = float(timing.get("worker_post_s", 0.0))
        post_s = max(0.0, worker_post_s) + max(0.0, ui_post_s)
        total_s = max(0.0, asr_s) + post_s
        return f"Timing: ASR {asr_s:.2f}s | Post {post_s:.2f}s | Total {total_s:.2f}s"

    def _set_timing_text(self, text: str) -> None:
        try:
            if hasattr(self.window, "timing_label") and self.window.timing_label is not None:
                self.window.timing_label.setText(str(text))
        except Exception:
            pass

    def _clear_cuda_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def _release_finished_workers(self) -> None:
        for attr in ("worker", "free_worker"):
            try:
                w = getattr(self.window, attr, None)
                if w is not None and not w.isRunning():
                    setattr(self.window, attr, None)
            except Exception:
                pass

    def unload_model(self) -> None:
        self.model = None
        self._release_finished_workers()
        gc.collect()
        self._clear_cuda_cache()

    # ------------------------ metrics helpers ----------------------

    @staticmethod
    def _clean_text_for_metrics(text: str) -> str:
        # Keep consistent with worker normalization
        return TranscribeWorker.clean_text(text)

    def _compute_cer(self, ref_text: str, hyp_text: str) -> float:
        ref = self._clean_text_for_metrics(ref_text).replace(" ", "")
        hyp = self._clean_text_for_metrics(hyp_text).replace(" ", "")
        try:
            return float(jiwer_cer(ref, hyp))
        except Exception:
            return 0.0

    @staticmethod
    def _norm_conf_from_logprob(lp: Optional[float]) -> Optional[float]:
        if lp is None:
            return None
        # Whisper avg_logprob is typically in [-1, 0]; map to [0, 1].
        conf = (float(lp) + 1.0) / 1.0
        return max(0.0, min(1.0, conf))

    def _augment_segments_and_fluency(
        self, segments: List[dict], hyp_text: str
    ) -> Tuple[List[dict], float, float, float, Optional[float]]:
        """
        Add duration, pause_before, conf to segments.
        Return (segments_aug, articulation_rate_wpm, pause_ratio,
                filled_pauses_count, avg_conf)
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
                max(
                    0.0,
                    float(segments[-1].get("end", 0.0))
                    - float(segments[0].get("start", 0.0)),
                )
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

    # ------------------------ error spans --------------------------

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

    # ------------------------ workflows ----------------------------

    def transcribe_and_score(self) -> None:
        """
        Script-based scoring. Uses ASR, computes WER/CER/Clarity and
        extended metrics when timestamps are available.
        """
        if self.window.current_audio_path is None:
            self.window.metrics_label.setText("No audio to score.")
            self._set_timing_text("Timing: -")
            return

        if self.window.free_speak_mode:
            self.transcribe_free()
            return

        self.window.btn_score.setEnabled(False)
        self.window.metrics_label.setText("Scoring... please wait")
        self._set_timing_text("Timing: running...")
        try:
            self.ensure_model()
        except Exception as e:
            self.window.metrics_label.setText(f"Scoring failed: {e}")
            self._set_timing_text("Timing: failed")
            self.window.btn_score.setEnabled(True)
            return

        self.window.worker = TranscribeWorker(
            self.model,
            self.window.current_script_text,
            self.window.current_audio_path,
            self.window,
            options=self.get_whisper_options(),
        )
        self._received_segments_this_run = False

        # Connect before start to avoid race
        try:
            self.window.worker.completed_with_segments.connect(
                self.on_transcription_done_with_segments
            )
        except Exception:
            pass
        self.window.worker.completed.connect(self.on_transcription_done)
        self.window.worker.failed.connect(self.on_worker_failed)
        self.window.worker.start()

    def transcribe_free(self) -> None:
        """
        Free Speak: no WER/CER; still compute fluency/confidence
        when timestamps are available.
        """
        if (
            self.window.audio_data is None
            and self.window.current_audio_path is None
        ):
            self.window.metrics_label.setText("Nothing to transcribe.")
            self._set_timing_text("Timing: -")
            return

        self.window.btn_score.setEnabled(False)
        self.window.metrics_label.setText("Transcribing... (Free Speak)")
        self._set_timing_text("Timing: running...")

        audio_input = (
            self.window.audio_data
            if self.window.audio_data is not None
            else self.window.current_audio_path
        )

        try:
            self.ensure_model()
        except Exception as e:
            self.window.metrics_label.setText(f"Transcription failed: {e}")
            self._set_timing_text("Timing: failed")
            self.window.btn_score.setEnabled(True)
            try:
                self.window.btn_save.setEnabled(False)
            except Exception:
                pass
            return

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
        self.window.free_worker.failed.connect(self.on_free_worker_failed)
        self.window.free_worker.start()

    # ------------------------ slots: scoring -----------------------

    @QtCore.pyqtSlot(str, float, float, float)
    def on_transcription_done(
        self, hyp: str, err: float, clar: float, score: float
    ) -> None:
        """
        Completion without segments. Show WER/CER/Clarity/Score,
        compute error spans (no timing metrics available).
        """
        t_ui0 = time.perf_counter()
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

            # Persist if session exists
            if getattr(self.window, "current_session_id", None) is not None:
                sess = db.update_session_scores(
                    self.window.db,
                    self.window.current_session_id,
                    hyp,
                    err,
                    clar,
                    score,
                    segments_json=None,
                    cer=cer_val,
                )
            else:
                # Create session if none (rare path)
                sess = db.add_session(
                    self.window.db,
                    self.window.current_script_name,
                    self.window.current_script_text,
                    self.window.current_audio_path,
                    hyp,
                    err,
                    clar,
                    score,
                    segments_json=None,
                    cer=cer_val,
                )
                self.window.current_session_id = sess.id
                formatted_timestamp = datetime.strptime(
                    sess.timestamp, "%Y-%m-%dT%H:%M:%S"
                ).strftime("%d %b %Y %H:%M")
                label = f"{formatted_timestamp} - {sess.script_name}"
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
            try:
                if sess is not None:
                    events = extract_error_events(
                        self.window.current_script_text, hyp
                    )
                    db.replace_session_errors(
                        self.window.db,
                        sess.id,
                        sess.timestamp,
                        sess.script_name,
                        events,
                    )
            except Exception:
                pass

            timing = self._get_worker_timing(free_speak=False)
            timing_text = self._format_timing_text(
                timing, time.perf_counter() - t_ui0
            )
            self.window.metrics_label.setText(
                f"Score: {score:.2f}/5 | WER: {err:.2%} | "
                f"CER: {cer_val:.2%} | Clarity: {clar:.2%}"
            )
            self._set_timing_text(timing_text)

        self.window.btn_score.setEnabled(True)
        self._release_finished_workers()

    @QtCore.pyqtSlot(str, float, float, float, object)
    def on_transcription_done_with_segments(
        self, hyp: str, err: float, clar: float, score: float, segments: object
    ) -> None:
        """
        Completion with segments: compute extended metrics + spans,
        update UI, and persist.
        """
        t_ui0 = time.perf_counter()
        try:
            self._received_segments_this_run = True

            seg_list = list(segments) if isinstance(segments, list) else []
            # Augment and compute extended metrics
            segs_aug, artic_rate, pause_ratio, filled_cnt, avg_conf = (
                self._augment_segments_and_fluency(seg_list, hyp)
            )
            cer_val = self._compute_cer(
                self.window.current_script_text, hyp
            )

            # Build transcript display and time ranges
            txt, segs_built, ranges, active_idx = build_transcript_from_segments(
                segs_aug
            )
            self.window.transcript_segments = segs_aug
            self.window.transcript_segment_ranges = ranges
            self.window.transcript_active_index = active_idx
            self.window.transcript_txt.setPlainText(txt)

            # Error highlights (script vs built transcript text)
            s_spans, t_spans = self.compute_error_spans(
                self.window.current_script_text, txt
            )
            self.window.set_error_highlights(s_spans, t_spans)

            # Label with extended metrics
            avg_conf_txt = f"{avg_conf:.0%}" if avg_conf is not None else "-"
            self.window.metrics_label.setText(
                "Score: "
                f"{score:.2f}/5 | WER: {err:.2%} | CER: {cer_val:.2%} | "
                f"Clarity: {clar:.2%} | Rate: {artic_rate:.0f} wpm | "
                f"Pauses: {pause_ratio:.0%} | Conf: {avg_conf_txt}"
            )

            # Initial playhead highlight along with base selections
            self.window.transcript_active_index = -1
            self.window.transcript_active_index = highlight_transcript_at_time(
                self.window.transcript_txt,
                self.window.transcript_segment_ranges,
                0.0,
                self.window.transcript_active_index,
                base_selections=self.window.transcript_error_selections,
            )

            # Persist
            segments_json = json.dumps(segs_aug)
            if getattr(self.window, "current_session_id", None) is not None:
                sess = db.update_session_scores(
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
                formatted_timestamp = datetime.strptime(
                    sess.timestamp, "%Y-%m-%dT%H:%M:%S"
                ).strftime("%d %b %Y %H:%M")
                label = f"{formatted_timestamp} - {sess.script_name}"
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
            try:
                if sess is not None:
                    events = extract_error_events(
                        self.window.current_script_text, hyp
                    )
                    db.replace_session_errors(
                        self.window.db,
                        sess.id,
                        sess.timestamp,
                        sess.script_name,
                        events,
                    )
            except Exception:
                pass

            timing = self._get_worker_timing(free_speak=False)
            timing_text = self._format_timing_text(
                timing, time.perf_counter() - t_ui0
            )
            avg_conf_txt = f"{avg_conf:.0%}" if avg_conf is not None else "-"
            self.window.metrics_label.setText(
                "Score: "
                f"{score:.2f}/5 | WER: {err:.2%} | CER: {cer_val:.2%} | "
                f"Clarity: {clar:.2%} | Rate: {artic_rate:.0f} wpm | "
                f"Pauses: {pause_ratio:.0%} | Conf: {avg_conf_txt}"
            )
            self._set_timing_text(timing_text)

        except Exception:
            # Fallback: show plain hypothesis if anything goes wrong
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        finally:
            self.window.btn_score.setEnabled(True)
            self._release_finished_workers()

    # ------------------------ slots: free speak ---------------------

    @QtCore.pyqtSlot(str)
    def on_free_transcription_done(self, hyp: str) -> None:
        """
        Free Speak without segments. No WER/CER. Keep transcript visible.
        """
        t_ui0 = time.perf_counter()
        self.window.last_transcript_text = hyp

        if not self._received_segments_this_run:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
            timing = self._get_worker_timing(free_speak=True)
            timing_text = self._format_timing_text(
                timing, time.perf_counter() - t_ui0
            )
            self.window.metrics_label.setText(
                "Transcript ready (Free Speak)"
            )
            self._set_timing_text(timing_text)
        self.window.btn_score.setEnabled(True)
        self.window.btn_save.setEnabled(True)
        self._release_finished_workers()

    @QtCore.pyqtSlot(str, object)
    def on_free_transcription_done_with_segments(
        self, hyp: str, segments: object
    ) -> None:
        """
        Free Speak with segments: compute fluency/confidence only.
        """
        t_ui0 = time.perf_counter()
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
                f"{avg_conf:.0%}" if avg_conf is not None else "-"
            )
            self.window.metrics_label.setText(
                "Transcript ready (Free Speak) | "
                f"Rate: {artic_rate:.0f} wpm | "
                f"Pauses: {pause_ratio:.0%} | Conf: {avg_conf_txt}"
            )
            timing = self._get_worker_timing(free_speak=True)
            timing_text = self._format_timing_text(
                timing, time.perf_counter() - t_ui0
            )
            self._set_timing_text(timing_text)

            self.window.transcript_active_index = -1
            self.window.transcript_active_index = highlight_transcript_at_time(
                self.window.transcript_txt,
                self.window.transcript_segment_ranges,
                0.0,
                self.window.transcript_active_index,
                base_selections=self.window.transcript_error_selections,
            )
            # Persistence happens on Save in free speak mode
        except Exception:
            self.window.transcript_txt.setText(hyp)
            self._clear_transcript_sync()
        finally:
            self.window.btn_score.setEnabled(True)
            self.window.btn_save.setEnabled(True)
            self._release_finished_workers()

    # ------------------------ error handling -----------------------

    @QtCore.pyqtSlot(str)
    def on_worker_failed(self, msg: str) -> None:
        self.window.metrics_label.setText(f"Scoring failed: {msg}")
        self._set_timing_text("Timing: failed")
        self.window.btn_score.setEnabled(True)
        self._release_finished_workers()

    @QtCore.pyqtSlot(str)
    def on_free_worker_failed(self, msg: str) -> None:
        self.window.metrics_label.setText(f"Transcription failed: {msg}")
        self._set_timing_text("Timing: failed")
        self.window.btn_score.setEnabled(True)
        try:
            self.window.btn_save.setEnabled(False)
        except Exception:
            pass
        self._release_finished_workers()

    # ------------------------ utilities ----------------------------

    def _clear_transcript_sync(self) -> None:
        """
        Clear transcript sync but keep base error selections (if any) visible.
        """
        self.window.transcript_segments = None
        self.window.transcript_segment_ranges = []
        self.window.transcript_active_index = -1
        try:
            self.window.transcript_txt.setExtraSelections(
                list(self.window.transcript_error_selections or [])
            )
        except Exception:
            pass
