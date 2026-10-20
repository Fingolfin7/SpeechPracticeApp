"""TTS baseline pre-check for generated practice scripts.

A neutral text-to-speech voice reads each generated script and the learner's own
transcription provider transcribes that reading. A script word that comes back
wrong from a perfect reading (an invented name, an archaic spelling, a numeral, a
homophone) would be counted as the learner's miss no matter how they say it, so
the generating model is told which words failed and asked to replace them before
the script is saved. Whatever still fails afterwards is stored on the script and
shown to the learner. Practice targets are never rewritten away; if a target
itself fails, the learner is told the card may not be measurable.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError
from django.utils import timezone

from alignment_utils import _normalize_token_text, align_tokens, tokenize_with_spans
from practice.models import PracticeSettings
from practice.services.script_generation import (
    GeneratedLadderDraft,
    GeneratedScriptDraft,
    get_script_generation_provider,
    revise_ladder_levels,
    revise_script_draft,
)
from practice.services.transcription import (
    LocalWhisperProvider,
    UploadedTranscriptProvider,
    get_transcription_provider,
)

logger = logging.getLogger(__name__)

PRECHECK_TTS_MODEL = "tts-1"
PRECHECK_TTS_VOICE = "alloy"
# The speech endpoint rejects longer inputs; longer scripts are read in pieces.
PRECHECK_TTS_MAX_CHARS = 4000
PRECHECK_MAX_WORKERS = 5
# Generation runs inside the request (gunicorn --timeout 120). Only ask the model
# for a rewrite when enough of that budget is left for a second model call.
PRECHECK_REVISION_CUTOFF_SECONDS = 55

STATUS_PASSED = "passed"
STATUS_FLAGGED = "flagged"
STATUS_SKIPPED = "skipped"


class PrecheckUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class WordMiss:
    word: str
    heard: str = ""

    def describe(self) -> str:
        if self.heard:
            return f'"{self.word}" (heard as "{self.heard}")'
        return f'"{self.word}" (dropped)'

    def as_dict(self) -> dict:
        return {"word": self.word, "heard": self.heard}


@dataclass(frozen=True)
class TargetMiss:
    miss: WordMiss
    failed: int
    total: int

    @property
    def mostly_failed(self) -> bool:
        return self.total > 0 and self.failed * 2 >= self.total


@dataclass(frozen=True)
class PrecheckResult:
    transcript: str
    misses: tuple[WordMiss, ...]
    text: str = ""

    def as_dict(self, revised: bool = False) -> dict:
        return {
            "status": STATUS_FLAGGED if self.misses else STATUS_PASSED,
            "misses": [miss.as_dict() for miss in self.misses],
            "transcript": self.transcript,
            "revised": revised,
            "tts_model": PRECHECK_TTS_MODEL,
            "voice": PRECHECK_TTS_VOICE,
            "checked_at": timezone.now().isoformat(),
        }


@dataclass
class PrecheckReport:
    """Outcome for one generated drill or ladder.

    Results are keyed by position in the draft (ladder level index, 0 for a drill).
    """

    status: str
    reason: str = ""
    results: dict[int, PrecheckResult] = field(default_factory=dict)
    # Non-target misses from the first reading: what the model was asked to fix.
    initial_misses: dict[int, tuple[WordMiss, ...]] = field(default_factory=dict)
    revised: tuple[int, ...] = ()
    revision_prompt: str = ""
    # Why no rewrite happened (or why it failed), when misses were found.
    revision_note: str = ""

    def stored(self, key: int, body: str) -> dict:
        """JSON for PracticeScript.precheck, tied to the body it checked."""
        result = self.results.get(key)
        if result is None:
            return {}
        data = result.as_dict(revised=key in self.revised)
        data["body_digest"] = hashlib.sha1((body or "").encode("utf-8")).hexdigest()[:16]
        return data

    def remaining(self, targets: list[str]) -> dict[int, tuple[WordMiss, ...]]:
        return _misses_to_fix(self.results, targets)

    def target_misses(self, targets: list[str]) -> list[TargetMiss]:
        """Practice targets the perfect reading got wrong, with how often across all scripts."""
        target_keys = _target_keys(targets)
        found: dict[str, WordMiss] = {}
        counts: dict[str, list[int]] = {}
        for result in self.results.values():
            for miss in result.misses:
                key = _normalize_token_text(miss.word)
                if key in target_keys:
                    found.setdefault(key, miss)
            for norm, failed in _word_outcomes(result.text, result.transcript):
                if norm in target_keys:
                    tally = counts.setdefault(norm, [0, 0])
                    tally[0] += int(failed)
                    tally[1] += 1
        return [
            TargetMiss(miss=miss, failed=counts.get(key, [1, 1])[0], total=counts.get(key, [1, 1])[1])
            for key, miss in found.items()
        ]

    def fixed_misses(self, targets: list[str]) -> list[WordMiss]:
        remaining = {
            _normalize_token_text(miss.word)
            for misses in self.remaining(targets).values()
            for miss in misses
        }
        fixed: dict[str, WordMiss] = {}
        for key in self.revised:
            for miss in self.initial_misses.get(key, ()):
                norm = _normalize_token_text(miss.word)
                if norm not in remaining:
                    fixed.setdefault(norm, miss)
        return list(fixed.values())


def precheck_enabled(user) -> bool:
    app_settings = _practice_settings(user)
    return bool(app_settings.tts_precheck_enabled) if app_settings is not None else True


def find_baseline_misses(script_text: str, transcript: str) -> tuple[WordMiss, ...]:
    """Script words a perfect reading did not reproduce, once each, in script order.

    Uses the same token normalization and join/split tolerance as scoring
    (alignment_utils), so a word listed here is one scoring would count as a miss.
    Insertions are ignored: they cannot be pinned on a script word.
    """
    script_text = script_text or ""
    transcript = transcript or ""
    ref = tokenize_with_spans(script_text)
    hyp = tokenize_with_spans(transcript)
    misses: list[WordMiss] = []
    seen: set[str] = set()
    for op, ref_index, hyp_index in align_tokens([token[0] for token in ref], [token[0] for token in hyp]):
        if op not in ("sub", "del") or ref_index is None:
            continue
        norm, start, end = ref[ref_index]
        if norm in seen:
            continue
        seen.add(norm)
        heard = ""
        if op == "sub" and hyp_index is not None:
            _hyp_norm, hyp_start, hyp_end = hyp[hyp_index]
            heard = transcript[hyp_start:hyp_end]
        misses.append(WordMiss(word=script_text[start:end], heard=heard))
    return tuple(misses)


def _word_outcomes(script_text: str, transcript: str) -> list[tuple[str, bool]]:
    """(normalized word, failed) for every script token, in order."""
    ref = [token[0] for token in tokenize_with_spans(script_text or "")]
    hyp = [token[0] for token in tokenize_with_spans(transcript or "")]
    outcomes = []
    for op, ref_index, _hyp_index in align_tokens(ref, hyp):
        if ref_index is None:
            continue
        if op == "equal_join_ref":
            # Two script words matched one heard word; both came through.
            outcomes.extend([(ref[ref_index], False), (ref[ref_index + 1], False)])
        else:
            outcomes.append((ref[ref_index], op in ("sub", "del")))
    return outcomes


class BaselineReader:
    """Reads text with TTS and transcribes it with the learner's own recognizer."""

    def __init__(self, user) -> None:
        app_settings = _practice_settings(user)
        api_key = (app_settings.get_secret("openai_api_key") if app_settings else None) or settings.OPENAI_API_KEY
        if not api_key:
            raise PrecheckUnavailable("it needs an OpenAI API key (Account page) for the text-to-speech reading")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PrecheckUnavailable("the 'openai' package is not installed") from exc
        transcriber = get_transcription_provider(user=user)
        if isinstance(transcriber, UploadedTranscriptProvider):
            raise PrecheckUnavailable("the transcript-sidecar provider cannot transcribe audio")
        self.client = OpenAI(api_key=api_key)
        self.transcriber = transcriber
        # Local Whisper shares one loaded model; do not run it concurrently.
        self.max_workers = 1 if isinstance(transcriber, LocalWhisperProvider) else PRECHECK_MAX_WORKERS

    def check(self, text: str) -> PrecheckResult:
        transcripts = []
        with tempfile.TemporaryDirectory(prefix="speechpractice-precheck-") as temp_dir:
            for index, piece in enumerate(_tts_pieces(text)):
                speech = self.client.audio.speech.create(
                    model=PRECHECK_TTS_MODEL,
                    voice=PRECHECK_TTS_VOICE,
                    input=piece,
                    response_format="mp3",
                )
                path = Path(temp_dir) / f"baseline-{index}.mp3"
                path.write_bytes(speech.read())
                transcripts.append(self.transcriber.transcribe(str(path)).text.strip())
        transcript = " ".join(part for part in transcripts if part)
        return PrecheckResult(transcript=transcript, misses=find_baseline_misses(text, transcript), text=text)

    def check_many(self, texts: dict[int, str]) -> dict[int, PrecheckResult]:
        """Check several texts in parallel; texts whose check errors are left out."""
        if not texts:
            return {}

        def run(item):
            key, text = item
            try:
                return key, self.check(text)
            except Exception:
                logger.warning("TTS pre-check failed for script %s", key, exc_info=True)
                return key, None

        workers = max(1, min(self.max_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, texts.items()))
        return {key: result for key, result in results if result is not None}


def precheck_ladder_draft(
    draft: GeneratedLadderDraft,
    *,
    user,
    targets: list[str],
    provider_name: str | None = None,
    started: float | None = None,
) -> tuple[GeneratedLadderDraft, PrecheckReport]:
    """Pre-check every level; ask the model to rewrite levels with failing words."""
    reader, report = _reader_or_report(user)
    if reader is None:
        return draft, report
    started = time.monotonic() if started is None else started
    levels = list(draft.levels)
    results = reader.check_many({index: level.body for index, level in enumerate(levels)})
    if not results:
        return draft, PrecheckReport(status=STATUS_SKIPPED, reason="the text-to-speech reading failed")
    report.results = results
    report.initial_misses = _misses_to_fix(results, targets)
    if report.initial_misses:
        index_by_level = {level.level: index for index, level in enumerate(levels)}
        if len(index_by_level) != len(levels):
            report.revision_note = "the model returned duplicate level numbers, so the levels could not be rewritten"
        elif _can_revise(report, provider_name, user, started):
            provider = get_script_generation_provider(provider_name, user=user)
            try:
                revisions, report.revision_prompt = revise_ladder_levels(
                    provider,
                    draft,
                    targets,
                    {levels[index].level: misses for index, misses in report.initial_misses.items()},
                )
            except Exception as exc:
                logger.warning("Ladder pre-check revision failed", exc_info=True)
                revisions, report.revision_note = [], f"the rewrite request failed ({exc})"
            if not revisions and not report.revision_note:
                report.revision_note = "the model did not return usable rewrites"
            candidates = {index_by_level[level.level]: level for level in revisions}
            rechecked = reader.check_many({index: level.body for index, level in candidates.items()})
            revised = []
            for index, level in candidates.items():
                new_result = rechecked.get(index)
                # Keep whichever version fails fewer words.
                if new_result is None or len(_non_target(new_result.misses, targets)) > len(
                    report.initial_misses[index]
                ):
                    continue
                levels[index] = level
                results[index] = new_result
                revised.append(index)
            report.revised = tuple(sorted(revised))
            if revised:
                draft = replace(draft, levels=tuple(levels))
    report.status = STATUS_FLAGGED if any(result.misses for result in results.values()) else STATUS_PASSED
    return draft, report


def precheck_script_draft(
    draft: GeneratedScriptDraft,
    *,
    user,
    targets: list[str],
    provider_name: str | None = None,
    started: float | None = None,
) -> tuple[GeneratedScriptDraft, PrecheckReport]:
    """Pre-check a drill; ask the model to rewrite it if words fail."""
    reader, report = _reader_or_report(user)
    if reader is None:
        return draft, report
    started = time.monotonic() if started is None else started
    results = reader.check_many({0: draft.body})
    if 0 not in results:
        return draft, PrecheckReport(status=STATUS_SKIPPED, reason="the text-to-speech reading failed")
    report.results = results
    report.initial_misses = _misses_to_fix(results, targets)
    if report.initial_misses and _can_revise(report, provider_name, user, started):
        provider = get_script_generation_provider(provider_name, user=user)
        misses = report.initial_misses[0]
        try:
            revision, report.revision_prompt = revise_script_draft(provider, draft, targets, misses)
        except Exception as exc:
            logger.warning("Drill pre-check revision failed", exc_info=True)
            revision, report.revision_note = None, f"the rewrite request failed ({exc})"
        if revision is None and not report.revision_note:
            report.revision_note = "the model did not return a usable rewrite"
        if revision is not None:
            new_result = reader.check_many({0: revision.body}).get(0)
            if new_result is not None and len(_non_target(new_result.misses, targets)) <= len(misses):
                draft = revision
                results[0] = new_result
                report.revised = (0,)
    report.status = STATUS_FLAGGED if results[0].misses else STATUS_PASSED
    return draft, report


def _reader_or_report(user) -> tuple[BaselineReader | None, PrecheckReport]:
    if not precheck_enabled(user):
        return None, PrecheckReport(status=STATUS_SKIPPED, reason="disabled")
    try:
        return BaselineReader(user), PrecheckReport(status=STATUS_PASSED)
    except PrecheckUnavailable as exc:
        return None, PrecheckReport(status=STATUS_SKIPPED, reason=str(exc))


def _can_revise(report: PrecheckReport, provider_name: str | None, user, started: float) -> bool:
    provider = get_script_generation_provider(provider_name, user=user)
    if not hasattr(provider, "complete"):
        report.revision_note = "the local template cannot rewrite scripts; pick an AI model to have them replaced"
        return False
    if time.monotonic() - started >= PRECHECK_REVISION_CUTOFF_SECONDS:
        report.revision_note = "generation took too long to also ask the model for a rewrite"
        return False
    return True


def _misses_to_fix(results: dict[int, PrecheckResult], targets: list[str]) -> dict[int, tuple[WordMiss, ...]]:
    # Practice targets stay in the script even when they fail; the learner is told instead.
    to_fix = {key: _non_target(result.misses, targets) for key, result in results.items()}
    return {key: misses for key, misses in to_fix.items() if misses}


def _target_keys(targets: list[str]) -> set[str]:
    keys = set()
    for target in targets:
        for norm, _start, _end in tokenize_with_spans(target or ""):
            keys.add(norm)
    return keys


def _non_target(misses, targets: list[str]) -> tuple[WordMiss, ...]:
    target_keys = _target_keys(targets)
    return tuple(miss for miss in misses if _normalize_token_text(miss.word) not in target_keys)


def _tts_pieces(text: str) -> list[str]:
    text = (text or "").strip()
    if len(text) <= PRECHECK_TTS_MAX_CHARS:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > PRECHECK_TTS_MAX_CHARS and current:
            pieces.append(current)
            candidate = line
        current = candidate[:PRECHECK_TTS_MAX_CHARS]
    if current.strip():
        pieces.append(current)
    return pieces


def _practice_settings(user=None) -> PracticeSettings | None:
    try:
        return PracticeSettings.load(user)
    except (OperationalError, ProgrammingError, ValueError):
        return None
