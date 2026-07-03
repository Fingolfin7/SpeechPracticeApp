from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from error_analytics import phrase_trend_summary
from django.contrib.auth import get_user_model
from django.db import connection
from django.db.utils import OperationalError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import (
    GeneratedPracticeScript,
    ImprovementCard,
    PracticeLadder,
    PracticeLadderStep,
    PracticeReview,
    PracticeScript,
    PracticeSettings,
    PracticeSession,
    ScoringJob,
    SessionError,
)
from .context_processors import static_version
from .services.jobs import create_scoring_job, process_scoring_job
from .services.scoring import score_transcript
from .services.script_import import import_script_items, parse_csv_text, parse_json_text
from .services.script_generation import (
    _stream_response_text,
    generate_local_template,
    parse_generated_ladder,
    parse_generated_script,
    script_generation_provider_choices,
)
from .services.codex_auth import serialize_token_bundle
from .services.analytics import build_card_candidates, refresh_improvement_cards, today_queue
from .services.transcription import OpenAITranscriptionProvider, TranscriptResult, UploadedTranscriptProvider


class PracticeWebTests(TransactionTestCase):
    def setUp(self):
        self.user, _created = get_user_model().objects.get_or_create(
            username="owner",
            defaults={"is_staff": True, "is_superuser": True},
        )
        self.user.set_password("test-pass-owner")
        self.user.is_staff = True
        self.user.is_superuser = True
        self.user.save()
        self.client.force_login(self.user)

    def _write_test_wav(self, path: Path, duration_seconds: float = 0.25) -> None:
        sample_rate = 8000
        frame_count = int(sample_rate * duration_seconds)
        silence = b"\x00\x00" * frame_count
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(silence)

    def test_static_version_uses_app_css_mtime(self):
        version = static_version(None)["static_version"]

        expected_keys = {
            "account_js",
            "app",
            "favicon",
            "history_js",
            "job_status_js",
            "progress_js",
            "recorder_js",
        }
        self.assertTrue(expected_keys.issubset(version))
        for key in expected_keys:
            self.assertGreater(version[key], 0, key)

    def test_script_library_page_renders(self):
        PracticeScript.objects.create(
            title="Breath Reading",
            body="A clean phrase begins with a calm breath.",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_USER,
        )
        PracticeScript.objects.create(
            title="Uploaded Reading",
            body="A user uploaded line.",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_UPLOADED,
        )
        PracticeScript.objects.create(
            title="Generated Drill",
            body="A generated focus line.",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )

        response = self.client.get(reverse("practice:scripts"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Reading scripts")
        self.assertContains(response, "Ladders / Drills")
        self.assertContains(response, "Breath Reading")
        self.assertContains(response, "User uploaded readings")
        self.assertContains(response, "Uploaded Reading")
        self.assertNotContains(response, "Generated Drill")
        self.assertContains(response, "Edit")
        self.assertContains(response, "Delete")

        drill_response = self.client.get(reverse("practice:scripts"), {"kind": "drill"})
        self.assertContains(drill_response, "Practice ladders")
        self.assertContains(drill_response, "Generate ladder")
        self.assertContains(drill_response, "AI generated drills")
        self.assertContains(drill_response, "Generated Drill")
        self.assertNotContains(drill_response, "Breath Reading")

    def test_script_edit_view_updates_script(self):
        script = PracticeScript.objects.create(
            title="Breath Drill",
            body="A clean phrase begins with a calm breath.",
            source=PracticeScript.SOURCE_USER,
            tags=["breath"],
        )

        response = self.client.post(
            reverse("practice:script_edit", args=[script.pk]),
            {
                "title": "Updated Breath Drill",
                "author": "Coach",
                "body": "A calmer phrase begins with a better breath.",
                "practice_kind": PracticeScript.KIND_READING,
                "source": PracticeScript.SOURCE_USER,
                "difficulty": 2,
                "active": "on",
                "tags_text": "breath,updated",
            },
        )

        self.assertEqual(response.status_code, 302)
        script.refresh_from_db()
        self.assertEqual(script.title, "Updated Breath Drill")
        self.assertEqual(script.author, "Coach")
        self.assertEqual(script.difficulty, 2)
        self.assertEqual(script.tags, ["breath", "updated"])

    def test_script_delete_view_removes_script(self):
        script = PracticeScript.objects.create(
            title="Delete Drill",
            body="This script can go.",
            source=PracticeScript.SOURCE_USER,
        )

        response = self.client.post(reverse("practice:script_delete", args=[script.pk]))

        self.assertEqual(response.status_code, 302)
        self.assertFalse(PracticeScript.objects.filter(pk=script.pk).exists())

    def test_script_bulk_delete_removes_selected_but_keeps_builtin(self):
        first = PracticeScript.objects.create(
            title="Bulk One",
            body="first bulk script",
            source=PracticeScript.SOURCE_USER,
        )
        second = PracticeScript.objects.create(
            title="Bulk Two",
            body="second bulk script",
            source=PracticeScript.SOURCE_UPLOADED,
        )
        builtin = PracticeScript.objects.create(
            title="Builtin Keeper",
            body="builtin script",
            source=PracticeScript.SOURCE_BUILTIN,
        )
        kept = PracticeScript.objects.create(
            title="Not Selected",
            body="unselected script",
            source=PracticeScript.SOURCE_USER,
        )

        response = self.client.post(
            reverse("practice:script_bulk_delete"),
            {"selected": [first.pk, second.pk, builtin.pk], "kind": "reading"},
        )

        self.assertRedirects(response, f"{reverse('practice:scripts')}?kind=reading")
        self.assertFalse(PracticeScript.objects.filter(pk__in=[first.pk, second.pk]).exists())
        self.assertTrue(PracticeScript.objects.filter(pk=builtin.pk).exists())
        self.assertTrue(PracticeScript.objects.filter(pk=kept.pk).exists())

    def test_card_bulk_delete_removes_selected_cards(self):
        first = ImprovementCard.objects.create(
            title="Word focus: bulk",
            kind=ImprovementCard.KIND_WORD,
            target_key="bulk",
        )
        second = ImprovementCard.objects.create(
            title="Word focus: batch",
            kind=ImprovementCard.KIND_WORD,
            target_key="batch",
        )
        kept = ImprovementCard.objects.create(
            title="Word focus: keep",
            kind=ImprovementCard.KIND_WORD,
            target_key="keep",
        )

        response = self.client.post(
            reverse("practice:card_bulk_delete"),
            {"selected": [first.pk, second.pk]},
        )

        self.assertRedirects(response, reverse("practice:cards"))
        self.assertFalse(ImprovementCard.objects.filter(pk__in=[first.pk, second.pk]).exists())
        self.assertTrue(ImprovementCard.objects.filter(pk=kept.pk).exists())

    def test_ladder_bulk_delete_skips_builtin_and_removes_generated_scripts(self):
        generated_script = PracticeScript.objects.create(
            title="Bulk Ladder Level 1",
            body="level one line",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
            source_ref="ladder:placeholder:level:1",
        )
        generated = PracticeLadder.objects.create(
            title="Bulk Generated Ladder",
            source=PracticeLadder.SOURCE_GENERATED,
        )
        generated_script.source_ref = f"ladder:{generated.pk}:level:1"
        generated_script.save(update_fields=["source_ref", "updated_at"])
        PracticeLadderStep.objects.create(
            ladder=generated,
            script=generated_script,
            level=1,
            title="Level 1",
        )
        builtin = PracticeLadder.objects.create(
            title="Bulk Builtin Ladder",
            source=PracticeLadder.SOURCE_BUILTIN,
        )

        response = self.client.post(
            reverse("practice:ladder_bulk_delete"),
            {"selected": [generated.pk, builtin.pk]},
        )

        self.assertRedirects(response, f"{reverse('practice:scripts')}?kind=drill")
        self.assertFalse(PracticeLadder.objects.filter(pk=generated.pk).exists())
        self.assertFalse(PracticeScript.objects.filter(pk=generated_script.pk).exists())
        self.assertTrue(PracticeLadder.objects.filter(pk=builtin.pk).exists())

    def test_session_bulk_delete_removes_sessions_audio_and_related_rows(self):
        self._create_legacy_tables()
        with tempfile.TemporaryDirectory() as temp_dir:
            first_audio = Path(temp_dir) / "bulk-one.webm"
            first_audio.write_bytes(b"audio-one")
            second_audio = Path(temp_dir) / "bulk-two.webm"
            second_audio.write_bytes(b"audio-two")
            first = PracticeSession.objects.create(
                timestamp="2026-06-14T12:00:00",
                script_name="Bulk Session One",
                script_text="clear practice text",
                audio_path=str(first_audio),
                transcript="clear practice text",
            )
            second = PracticeSession.objects.create(
                timestamp="2026-06-15T12:00:00",
                script_name="Bulk Session Two",
                script_text="clear practice text",
                audio_path=str(second_audio),
                transcript="clear practice text",
            )
            kept = PracticeSession.objects.create(
                timestamp="2026-06-16T12:00:00",
                script_name="Kept Session",
                script_text="clear practice text",
                audio_path="",
                transcript="clear practice text",
            )
            SessionError.objects.create(
                session_id=first.pk,
                timestamp=first.timestamp,
                script_name=first.script_name,
                ref_token="practice",
                op="del",
                error_kind="word_missing",
            )
            job = ScoringJob.objects.create(
                script_name=first.script_name,
                script_text=first.script_text,
                audio_path=str(first_audio),
                provider="uploaded_transcript",
                legacy_session_id=first.pk,
            )

            response = self.client.post(
                reverse("practice:session_bulk_delete"),
                {"selected": [first.pk, second.pk]},
            )

            self.assertRedirects(response, reverse("practice:sessions"))
            self.assertFalse(first_audio.exists())
            self.assertFalse(second_audio.exists())
            self.assertFalse(PracticeSession.objects.filter(pk__in=[first.pk, second.pk]).exists())
            self.assertTrue(PracticeSession.objects.filter(pk=kept.pk).exists())
            self.assertEqual(SessionError.objects.filter(session_id=first.pk).count(), 0)
            job.refresh_from_db()
            self.assertIsNone(job.legacy_session_id)

    def test_today_queue_pairs_due_card_with_latest_generated_script(self):
        due_card = ImprovementCard.objects.create(
            title="Word focus: ready",
            kind=ImprovementCard.KIND_WORD,
            target_key="ready",
            prompt="Practice ready.",
            mastery=0.4,
            due_at=timezone.now() - timezone.timedelta(hours=1),
        )
        script = PracticeScript.objects.create(
            title="Drill: ready",
            body="ready words rise",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=due_card, script=script)

        queue = today_queue(limit=1)

        self.assertEqual(queue[0].card, due_card)
        self.assertEqual(queue[0].script, script)
        self.assertTrue(queue[0].is_due)

    def test_dashboard_renders_today_queue(self):
        self._create_legacy_tables()
        card = ImprovementCard.objects.create(
            title="Sound pattern: S",
            kind=ImprovementCard.KIND_SOUND,
            target_key="S",
            prompt="Practice S.",
            mastery=0.5,
            due_at=timezone.now() - timezone.timedelta(minutes=5),
        )
        script = PracticeScript.objects.create(
            title="Drill: S",
            body="soft sounds stay steady",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=script)

        response = self.client.get(reverse("practice:dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Next reps")
        self.assertContains(response, "Sound")
        self.assertContains(response, "S")
        self.assertContains(response, "Practice drill")
        self.assertContains(response, "Do next")
        # A single queue item is featured in the hero, not repeated in the list.
        self.assertContains(response, "This card is due now")
        self.assertContains(response, "Your next rep is featured above")

    def test_dashboard_queue_list_continues_after_hero(self):
        self._create_legacy_tables()
        now = timezone.now()
        for index in range(3):
            ImprovementCard.objects.create(
                title=f"Sound pattern: S{index}",
                kind=ImprovementCard.KIND_SOUND,
                target_key=f"S{index}",
                prompt=f"Practice S{index}.",
                mastery=0.5,
                due_at=now - timezone.timedelta(minutes=5 + index),
            )

        response = self.client.get(reverse("practice:dashboard"))

        self.assertEqual(response.status_code, 200)
        # The hero takes the first card; the list resumes at #2.
        self.assertContains(response, "Why this is next")
        self.assertContains(response, '<div class="today-index">2</div>', html=False)
        self.assertNotContains(response, '<div class="today-index">1</div>', html=False)

    def test_poetry_foundation_csv_import_shape(self):
        csv_text = (
            ",Title,Poem,Poet,Tags\n"
            '0,"  Small Song  ","First line\\nSecond line","Ada Poet","breath,poem"\n'
        )
        items = parse_csv_text(csv_text, source_ref="poems.csv")

        result = import_script_items(items, extra_tags=["csv"])

        self.assertEqual(result.created, 1)
        script = PracticeScript.objects.get(title="Small Song")
        self.assertEqual(script.author, "Ada Poet")
        self.assertIn("First line", script.body)
        self.assertIn("poem", script.tags)
        self.assertIn("csv", script.tags)

    def test_json_title_mapping_import_shape(self):
        items = parse_json_text(
            '{"Quiet Poem": "A quiet line\\nA clearer line"}',
            source_ref="poems.json",
        )

        result = import_script_items(items)

        self.assertEqual(result.created, 1)
        self.assertTrue(PracticeScript.objects.filter(title="Quiet Poem").exists())

    def test_script_import_view_accepts_uploaded_text_file(self):
        upload = SimpleUploadedFile(
            "custom.txt",
            b"Title: Custom Drill\nAuthor: Test Poet\nSpeak clearly.",
            content_type="text/plain",
        )

        response = self.client.post(
            reverse("practice:script_import"),
            {
                "files": [upload],
                "default_author": "",
                "tags_text": "upload,test",
                "replace": "on",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Import result")
        script = PracticeScript.objects.get(title="Custom Drill")
        self.assertEqual(script.author, "Test Poet")
        self.assertEqual(script.practice_kind, PracticeScript.KIND_READING)
        self.assertEqual(script.source, PracticeScript.SOURCE_UPLOADED)
        self.assertIn("upload", script.tags)

    def test_card_script_generation_template(self):
        card = ImprovementCard.objects.create(
            title="Word focus: river",
            kind=ImprovementCard.KIND_WORD,
            target_key="river",
            prompt="Practice river in short phrases.",
            mastery=0.25,
        )

        draft = generate_local_template(card)

        self.assertEqual(draft.provider, "local_template")
        self.assertIn("river", draft.body)
        self.assertIn('the word "river"', draft.prompt_snapshot)
        self.assertIn("read-aloud drill", draft.prompt_snapshot)

    def test_phrase_script_generation_template_repeats_full_phrase(self):
        card = ImprovementCard.objects.create(
            title="Phrase focus: steady breath today",
            kind=ImprovementCard.KIND_PHRASE,
            target_key="steady breath today",
            prompt="Practice the phrase without rushing.",
            mastery=0.2,
        )

        draft = generate_local_template(card)

        self.assertEqual(draft.provider, "local_template")
        self.assertEqual(draft.body.count("steady breath today"), 8)

    def test_phrase_trend_summary_finds_error_context_phrase(self):
        scored = SimpleNamespace(
            id=1,
            script_name="Phrase Drill",
            script_text="keep steady breath today",
        )
        event = SimpleNamespace(
            session_id=1,
            ref_token="breath",
            hyp_token=None,
            op="del",
            error_kind="word_missing",
            ref_start=12,
            ref_end=18,
            ref_local_start=0,
            ref_local_end=6,
            ref_token_len=6,
        )

        summary = phrase_trend_summary([scored], [event], top_n=3, min_attempts=1)

        phrases = summary["top_trouble_phrases"]
        self.assertEqual(phrases[0]["phrase"], "steady breath today")
        self.assertEqual(phrases[0]["errors"], 1.0)

    def test_trend_summary_for_range_reads_django_orm_data(self):
        self._create_legacy_tables()
        now = timezone.now()
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
        scored = PracticeSession.objects.create(
            user=self.user,
            timestamp=timestamp,
            script_name="Phrase Drill",
            script_text="keep steady breath today keep steady breath today",
            audio_path="recording.wav",
            transcript="keep steady today keep steady breath today",
            wer=0.25,
            clarity=0.75,
            score=3.5,
        )
        SessionError.objects.create(
            user=self.user,
            session_id=scored.id,
            timestamp=timestamp,
            script_name=scored.script_name,
            ref_token="breath",
            hyp_token=None,
            op="del",
            error_kind="word_missing",
            ref_start=12,
            ref_end=18,
            ref_local_start=0,
            ref_local_end=6,
            ref_token_len=6,
        )
        # A char_replace event exercises the confusion path, which reads
        # hyp_local_* fields — kept in the fetch's .only() list.
        SessionError.objects.create(
            user=self.user,
            session_id=scored.id,
            timestamp=timestamp,
            script_name=scored.script_name,
            ref_token="keep",
            hyp_token="creep",
            op="sub",
            error_kind="char_replace",
            ref_start=0,
            ref_end=4,
            ref_local_start=0,
            ref_local_end=1,
            hyp_local_start=0,
            hyp_local_end=2,
            ref_token_len=4,
            hyp_token_len=5,
        )
        other_user = get_user_model().objects.create(username="someone-else")
        PracticeSession.objects.create(
            user=other_user,
            timestamp=timestamp,
            script_name="Other Drill",
            script_text="not your words",
            audio_path="other.wav",
            transcript="not your words",
        )

        from .services.analytics import trend_summary_for_range

        # One session query + one event query cover all five summaries; a
        # higher count means deferred-field loads crept back in.
        with self.assertNumQueries(2):
            summary = trend_summary_for_range(
                user=self.user,
                start_dt=(now - timezone.timedelta(days=7)).replace(tzinfo=None),
                end_dt=now.replace(tzinfo=None),
            )

        words = {row["word"] for row in summary["words"]["top_trouble_words"]}
        self.assertIn("breath", words)
        self.assertNotIn("your", words)
        phrases = {row["phrase"] for row in summary["phrases"]["top_trouble_phrases"]}
        self.assertIn("steady breath today", phrases)
        confusions = summary["characters"]["top_character_confusions"]
        self.assertEqual(confusions[0]["from"], "k")
        self.assertEqual(confusions[0]["to"], "cr")
        self.assertEqual(summary["words"]["recent_session_count"], 1)

    def test_build_card_candidates_includes_phrase_focus(self):
        candidates = build_card_candidates(
            {
                "words": {},
                "sounds": {},
                "positions": {},
                "phrases": {
                    "top_trouble_phrases": [
                        {
                            "phrase": "steady breath today",
                            "attempts": 2.0,
                            "errors": 1.0,
                            "error_rate": 0.5,
                        }
                    ]
                },
            }
        )

        self.assertEqual(candidates[0]["kind"], ImprovementCard.KIND_PHRASE)
        self.assertEqual(candidates[0]["target_key"], "steady breath today")
        self.assertEqual(candidates[0]["mastery"], 0.5)

    def test_refresh_improvement_cards_records_exact_source_range(self):
        summary = {
            "words": {
                "top_trouble_words": [
                    {
                        "word": "steady",
                        "error_rate": 0.5,
                        "errors": 2,
                        "attempts": 4,
                    }
                ]
            },
            "sounds": {},
            "phrases": {},
            "positions": {},
        }
        start_dt = timezone.datetime(2026, 6, 1)
        end_dt = timezone.datetime(2026, 6, 19, 23, 59, 59)

        with patch("practice.services.analytics.trend_summary_for_range", return_value=summary):
            created = refresh_improvement_cards(start_dt=start_dt, end_dt=end_dt)

        self.assertEqual(created, 1)
        card = ImprovementCard.objects.get(target_key="steady")
        self.assertEqual(card.stats["source_window_start"], "2026-06-01")
        self.assertEqual(card.stats["source_window_end"], "2026-06-19")
        self.assertEqual(card.stats["source_window_label"], "2026-06-01 to 2026-06-19")

    def test_generated_script_parser_handles_structured_output(self):
        title, body = parse_generated_script(
            "TITLE: Crisp TH Drill\nSCRIPT:\nThin things thrive.\nBreathe through the phrase.",
            fallback_title="Fallback",
        )

        self.assertEqual(title, "Crisp TH Drill")
        self.assertIn("Thin things thrive", body)
        self.assertNotIn("TITLE:", body)

    def test_generated_ladder_parser_handles_json_output(self):
        title, theme, levels = parse_generated_ladder(
            """
            {
              "title": "Rainy R Ladder",
              "theme": "R practice in a rainy noir scene",
              "levels": [
                {"level": 1, "title": "Warm-Up", "focus": ["R"], "lines": ["red rain", "round room"]},
                {"level": 2, "title": "Contrast", "focus": ["R"], "lines": ["rare river runs"]}
              ]
            }
            """,
            fallback_title="Fallback",
        )

        self.assertEqual(title, "Rainy R Ladder")
        self.assertIn("rainy noir", theme)
        self.assertEqual(levels[0].level, 1)
        self.assertIn("red rain", levels[0].body)
        self.assertEqual(levels[0].focus, ("R",))

    def test_codex_stream_response_text_collects_deltas(self):
        event_stream = iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta='{"title":"'),
                SimpleNamespace(type="response.output_text.delta", delta='Streamed Ladder"}'),
                SimpleNamespace(type="response.completed", response=None),
            ]
        )

        self.assertEqual(_stream_response_text(event_stream), '{"title":"Streamed Ladder"}')

    def test_script_generation_provider_choices_include_offline_default(self):
        choices = dict(script_generation_provider_choices())

        self.assertIn("local_template", choices)
        self.assertIn("openai", choices)
        self.assertIn("anthropic", choices)

    def test_score_transcript_for_matching_text(self):
        transcript = TranscriptResult(
            text="A clean phrase begins with a calm breath.",
            provider="test",
            segments=[],
            raw={},
        )

        result = score_transcript(
            "A clean phrase begins with a calm breath.",
            transcript,
        )

        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertGreater(result.score, 4.9)

    @override_settings(
        OPENAI_API_KEY="",
        CODEX_CHATGPT_BASE_URL="https://chatgpt.example.test/codex",
    )
    def test_openai_transcription_uses_codex_auth_without_api_key(self):
        settings_obj = PracticeSettings.load()
        settings_obj.openai_transcription_model = "whisper-1"
        settings_obj.set_secret(
            "codex_token_bundle",
            serialize_token_bundle(
                {
                    "access_token": "codex-access-token",
                    "refresh_token": "codex-refresh-token",
                    "id_token": "codex-id-token",
                }
            ),
        )
        settings_obj.save()
        calls = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.audio = SimpleNamespace(
                    transcriptions=SimpleNamespace(create=self.create),
                )

            def create(self, **kwargs):
                return SimpleNamespace(text=f"{kwargs['model']}: clear codex speech")

        fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "recording.wav"
            self._write_test_wav(audio_path)
            with patch.dict("sys.modules", {"openai": fake_openai_module}):
                result = OpenAITranscriptionProvider().transcribe(str(audio_path))

        self.assertEqual(result.text, "whisper-1: clear codex speech")
        self.assertEqual(result.raw["auth_source"], "codex")
        self.assertEqual(
            calls,
            [
                {"api_key": "codex-access-token"}
            ],
        )

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_DIRECT_UPLOAD_MAX_BYTES=0)
    def test_openai_whisper_transcription_chunks_partials_and_offsets_segments(self):
        settings_obj = PracticeSettings.load()
        settings_obj.openai_transcription_model = "whisper-1"
        settings_obj.whisper_chunk_seconds = 10
        settings_obj.save()
        create_calls = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.audio = SimpleNamespace(
                    transcriptions=SimpleNamespace(create=self.create),
                )

            def create(self, **kwargs):
                create_calls.append(kwargs)
                chunk_number = len(create_calls)
                return SimpleNamespace(
                    text=f"chunk {chunk_number}",
                    segments=[
                        {
                            "text": f"chunk {chunk_number}",
                            "start": 0.0,
                            "end": 0.5,
                            "avg_logprob": -0.1,
                        }
                    ],
                )

        partials = []
        fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "recording.wav"
            self._write_test_wav(audio_path, duration_seconds=21.2)
            with patch.dict("sys.modules", {"openai": fake_openai_module}):
                result = OpenAITranscriptionProvider().transcribe(
                    str(audio_path),
                    partial_callback=partials.append,
                )

        self.assertEqual(result.text, "chunk 1 chunk 2 chunk 3")
        self.assertEqual(partials, ["chunk 1", "chunk 1 chunk 2", "chunk 1 chunk 2 chunk 3"])
        self.assertEqual([segment["start"] for segment in result.segments], [0.0, 10.0, 20.0])
        self.assertEqual([segment["end"] for segment in result.segments], [0.5, 10.5, 20.5])
        self.assertEqual(
            [
                (
                    call["model"],
                    call["response_format"],
                    call["timestamp_granularities"],
                    call.get("prompt", ""),
                )
                for call in create_calls
            ],
            [
                ("whisper-1", "verbose_json", ["segment"], ""),
                ("whisper-1", "verbose_json", ["segment"], "chunk 1"),
                ("whisper-1", "verbose_json", ["segment"], "chunk 1 chunk 2"),
            ],
        )

    @override_settings(
        OPENAI_API_KEY="",
        CODEX_CHATGPT_BASE_URL="https://chatgpt.example.test/codex",
    )
    def test_openai_transcription_reports_missing_codex_api_scope_cleanly(self):
        settings_obj = PracticeSettings.load()
        settings_obj.set_secret(
            "codex_token_bundle",
            serialize_token_bundle(
                {
                    "access_token": "codex-access-token",
                    "refresh_token": "codex-refresh-token",
                    "id_token": "codex-id-token",
                }
            ),
        )
        settings_obj.save()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.audio = SimpleNamespace(
                    transcriptions=SimpleNamespace(create=self.create),
                )

            def create(self, **kwargs):
                raise RuntimeError(
                    "You have insufficient permissions for this operation. "
                    "Missing scopes: api.audio.write."
                )

        fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "recording.wav"
            self._write_test_wav(audio_path)
            with patch.dict("sys.modules", {"openai": fake_openai_module}):
                with self.assertRaisesRegex(RuntimeError, "does not include the API scopes"):
                    OpenAITranscriptionProvider().transcribe(str(audio_path))

    @override_settings(
        OPENAI_API_KEY="",
        CODEX_CHATGPT_BASE_URL="https://chatgpt.example.test/codex",
    )
    def test_openai_transcription_reports_codex_browser_challenge_cleanly(self):
        settings_obj = PracticeSettings.load()
        settings_obj.set_secret(
            "codex_token_bundle",
            serialize_token_bundle(
                {
                    "access_token": "codex-access-token",
                    "refresh_token": "codex-refresh-token",
                    "id_token": "codex-id-token",
                }
            ),
        )
        settings_obj.save()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.audio = SimpleNamespace(
                    transcriptions=SimpleNamespace(create=self.create),
                )

            def create(self, **kwargs):
                raise RuntimeError(
                    "<html><script src='/cdn-cgi/challenge-platform/test.js'></script>"
                )

        fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "recording.wav"
            self._write_test_wav(audio_path)
            with patch.dict("sys.modules", {"openai": fake_openai_module}):
                with self.assertRaisesRegex(RuntimeError, "browser challenge"):
                    OpenAITranscriptionProvider().transcribe(str(audio_path))

    def test_practice_page_renders_with_script(self):
        script = PracticeScript.objects.create(
            title="Morning Lines",
            body="The morning line is clear and deliberate.",
            source=PracticeScript.SOURCE_USER,
        )

        response = self.client.get(reverse("practice:practice"), {"script": script.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Morning Lines")
        self.assertContains(response, "Score recording")
        self.assertContains(response, "data-score-button disabled")
        self.assertContains(response, "Record or upload a take before scoring.")
        self.assertContains(response, "practice-task-tabs")
        self.assertContains(response, "Copy score")
        self.assertContains(response, "data-copy-practice-score")
        self.assertContains(response, 'data-script-preview-base="/scripts/0/preview/"')
        self.assertContains(response, "data-script-title")
        self.assertContains(response, "data-waveform-canvas")
        self.assertContains(response, "data-record-play")
        self.assertContains(response, "data-record-delete")
        self.assertContains(response, "Find a script")
        self.assertContains(response, "Random")
        self.assertContains(response, "Free Speak")
        self.assertContains(response, "Quick Practice")

    def test_practice_page_can_start_from_card_context(self):
        card = ImprovementCard.objects.create(
            title="Word focus: steady",
            kind=ImprovementCard.KIND_WORD,
            target_key="steady",
            prompt="Practice steady in short phrases.",
            mastery=0.35,
        )
        older_script = PracticeScript.objects.create(
            title="Older steady drill",
            body="steady starts here",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        latest_script = PracticeScript.objects.create(
            title="Latest steady drill",
            body="steady speech stays ready",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=older_script)
        GeneratedPracticeScript.objects.create(card=card, script=latest_script)

        response = self.client.get(reverse("practice:practice"), {"card": card.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "steady")
        self.assertContains(response, "Latest steady drill")
        self.assertContains(response, f'name="card" value="{card.pk}"')
        self.assertContains(response, 'value="quick" checked')

    def test_script_and_quick_modes_use_separate_script_kinds(self):
        reading = PracticeScript.objects.create(
            title="Reading Only",
            body="This is a normal reading passage.",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_USER,
        )
        drill = PracticeScript.objects.create(
            title="Drill Only",
            body="drill drill clearly",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )

        script_response = self.client.get(reverse("practice:practice"), {"script": reading.pk})
        self.assertContains(script_response, "Reading Only")
        self.assertContains(script_response, "Reading script:")
        self.assertNotContains(script_response, "Drill Only")

        quick_response = self.client.get(reverse("practice:practice"), {"mode": "quick", "script": drill.pk})
        self.assertContains(quick_response, "Drill Only")
        self.assertContains(quick_response, "Practice drill:")
        self.assertNotContains(quick_response, "Reading Only")

    def test_quick_practice_page_runs_selected_ladder_without_generation_controls(self):
        card = ImprovementCard.objects.create(
            title="Word focus: crisp",
            kind=ImprovementCard.KIND_WORD,
            target_key="crisp",
            prompt="Practice crisp endings.",
            mastery=0.25,
            due_at=timezone.now() - timezone.timedelta(minutes=10),
        )
        generated = PracticeScript.objects.create(
            title="Latest crisp drill",
            body="crisp clips click cleanly",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=generated)
        builtin = PracticeScript.objects.create(
            title="Tongue Twister Level 1: Warm-Up",
            body="Big brown bear.",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_BUILTIN,
            source_ref="builtin:speech-drills:tongue-twister-level-1",
            difficulty=1,
        )
        ladder = PracticeLadder.objects.create(
            title="Built-in Tongue Twister Ladder",
            theme="From warm-up to mastery",
            source=PracticeLadder.SOURCE_BUILTIN,
        )
        PracticeLadderStep.objects.create(
            ladder=ladder,
            script=builtin,
            level=1,
            title="Warm-Up",
            focus=["tongue-twister"],
        )

        response = self.client.get(reverse("practice:practice"), {"mode": "quick"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Run a drill ladder")
        self.assertContains(response, "Manage ladders")
        self.assertContains(response, "Drill ladder")
        self.assertContains(response, "data-ladder-select")
        self.assertContains(response, "Tongue Twister Level 1: Warm-Up")
        self.assertNotContains(response, "Generate drill")
        self.assertNotContains(response, "Generate ladder")

    def test_drill_library_shows_ladders_and_focus_card_picker(self):
        card = ImprovementCard.objects.create(
            title="Word focus: crisp",
            kind=ImprovementCard.KIND_WORD,
            target_key="crisp",
            prompt="Practice crisp endings.",
            mastery=0.25,
            due_at=timezone.now() - timezone.timedelta(minutes=10),
        )
        later_card = ImprovementCard.objects.create(
            title="Phrase focus: slow rain",
            kind=ImprovementCard.KIND_PHRASE,
            target_key="slow rain",
            prompt="Practice slow rain in phrases.",
            mastery=0.7,
            due_at=timezone.now() + timezone.timedelta(days=3),
        )
        script = PracticeScript.objects.create(
            title="Crisp Ladder Level 1",
            body="crisp clips click cleanly",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        ladder = PracticeLadder.objects.create(
            title="Crisp Courtroom Ladder",
            theme="Crisp endings in a courtroom drama.",
            source=PracticeLadder.SOURCE_GENERATED,
            model_provider="local_template",
            auth_source="local",
        )
        ladder.cards.add(card)
        PracticeLadderStep.objects.create(
            ladder=ladder,
            script=script,
            level=1,
            title="Opening statement",
            focus=["crisp"],
        )

        response = self.client.get(reverse("practice:scripts"), {"kind": "drill"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ladders / Drills")
        self.assertContains(response, "Use selected focus cards")
        self.assertContains(response, "crisp")
        self.assertContains(response, "slow rain")
        self.assertContains(response, "Crisp Courtroom Ladder")
        self.assertContains(response, "Crisp endings in a courtroom drama.")
        self.assertContains(response, 'name="cards" value="%s"' % card.pk)
        self.assertContains(response, 'name="cards" value="%s"' % later_card.pk)

    def test_generate_ladder_view_creates_steps_and_auth_metadata(self):
        cards = [
            ImprovementCard.objects.create(
                title=f"Sound pattern: R {idx}",
                kind=ImprovementCard.KIND_SOUND,
                target_key=f"R{idx}",
                prompt="Practice R without dropping endings.",
                mastery=0.2,
            )
            for idx in range(1, 6)
        ]

        response = self.client.post(
            reverse("practice:generate_ladder"),
            {
                "provider": "local_template",
                "theme": "courtroom drama",
                "cards": [str(card.pk) for card in cards],
            },
        )

        self.assertEqual(response.status_code, 302)
        ladder = PracticeLadder.objects.get(source=PracticeLadder.SOURCE_GENERATED)
        self.assertIn("courtroom drama", ladder.theme)
        self.assertEqual(ladder.auth_source, "local")
        self.assertEqual(ladder.cards.count(), 5)
        self.assertEqual(ladder.steps.count(), 5)
        self.assertEqual(PracticeScript.objects.filter(source_ref__startswith=f"ladder:{ladder.pk}:").count(), 5)
        self.assertEqual(
            GeneratedPracticeScript.objects.filter(card__in=cards, auth_source="local").count(),
            25,
        )
        self.assertIn(f"ladder={ladder.pk}", response["Location"])

    def test_delete_ladder_removes_generated_steps_and_scripts(self):
        script = PracticeScript.objects.create(
            title="Generated Ladder Level 1",
            body="level one line",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
            source_ref="ladder:placeholder:level:1",
        )
        ladder = PracticeLadder.objects.create(
            title="Delete Me Ladder",
            source=PracticeLadder.SOURCE_GENERATED,
        )
        script.source_ref = f"ladder:{ladder.pk}:level:1"
        script.save(update_fields=["source_ref", "updated_at"])
        PracticeLadderStep.objects.create(
            ladder=ladder,
            script=script,
            level=1,
            title="Level 1",
        )

        response = self.client.post(reverse("practice:delete_ladder", args=[ladder.pk]))

        self.assertRedirects(response, f"{reverse('practice:scripts')}?kind=drill")
        self.assertFalse(PracticeLadder.objects.filter(pk=ladder.pk).exists())
        self.assertFalse(PracticeScript.objects.filter(pk=script.pk).exists())

    def test_delete_ladder_blocks_builtin_ladder(self):
        script = PracticeScript.objects.create(
            title="Builtin Ladder Level 1",
            body="level one line",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_BUILTIN,
            source_ref="builtin:speech-drills:tongue-twister-level-1",
        )
        ladder = PracticeLadder.objects.create(
            title="Built-in Ladder",
            source=PracticeLadder.SOURCE_BUILTIN,
        )
        PracticeLadderStep.objects.create(
            ladder=ladder,
            script=script,
            level=1,
            title="Level 1",
        )

        response = self.client.post(reverse("practice:delete_ladder", args=[ladder.pk]))

        self.assertRedirects(response, f"{reverse('practice:scripts')}?kind=drill")
        self.assertTrue(PracticeLadder.objects.filter(pk=ladder.pk).exists())
        self.assertTrue(PracticeScript.objects.filter(pk=script.pk).exists())

    def test_script_preview_endpoint_returns_metadata(self):
        script = PracticeScript.objects.create(
            title="Sibilant Steps",
            author="Practice Lab",
            body="Soft sounds stay steady.\n\nSmall steps sharpen speech.",
            source=PracticeScript.SOURCE_IMPORTED,
            difficulty=3,
            tags=["sibilants", "phrases"],
        )

        response = self.client.get(reverse("practice:script_preview", args=[script.pk]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["title"], "Sibilant Steps")
        self.assertEqual(payload["author"], "Practice Lab")
        self.assertEqual(payload["word_count"], 8)
        self.assertEqual(payload["practice_kind"], "Reading script")
        self.assertEqual(payload["source"], "Imported")
        self.assertEqual(payload["difficulty"], 3)
        self.assertEqual(payload["tags"], ["sibilants", "phrases"])

    def test_script_preview_ignores_inactive_scripts(self):
        script = PracticeScript.objects.create(
            title="Dormant Drill",
            body="This one should not be selectable.",
            active=False,
        )

        response = self.client.get(reverse("practice:script_preview", args=[script.pk]))

        self.assertEqual(response.status_code, 404)

    def test_practice_script_mode_picks_random_reading_script_by_default(self):
        first = PracticeScript.objects.create(
            title="First Reading",
            body="first reading text",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_USER,
        )
        second = PracticeScript.objects.create(
            title="Second Reading",
            body="second reading text",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_USER,
        )

        with patch("practice.views.random.randrange", return_value=1):
            response = self.client.get(reverse("practice:practice"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["selected_script"], second)
        self.assertNotEqual(response.context["selected_script"], first)

    def test_practice_script_query_overrides_random_default(self):
        PracticeScript.objects.create(
            title="Other Reading",
            body="other reading text",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_USER,
        )
        selected = PracticeScript.objects.create(
            title="Chosen Reading",
            body="chosen reading text",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_USER,
        )

        with patch("practice.views.random.randrange", return_value=0):
            response = self.client.get(f"{reverse('practice:practice')}?script={selected.pk}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["selected_script"], selected)

    def test_uploaded_transcript_provider_reads_txt_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "recording.txt"
            path.write_text("clear practice text", encoding="utf-8")

            result = UploadedTranscriptProvider().transcribe(str(path))

        self.assertEqual(result.text, "clear practice text")
        self.assertEqual(result.provider, "uploaded_transcript")

    def test_scoring_job_processes_uploaded_transcript(self):
        self._create_legacy_tables()
        script = PracticeScript.objects.create(
            title="Plain Drill",
            body="clear practice text",
            source=PracticeScript.SOURCE_USER,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "recording.txt"
            path.write_text("clear practice text", encoding="utf-8")
            job = create_scoring_job(
                script=script,
                audio_path=str(path),
                provider="uploaded_transcript",
            )

            with patch("practice.services.jobs.refresh_improvement_cards") as refresh:
                processed = process_scoring_job(job.pk)

        self.assertEqual(processed.status, ScoringJob.STATUS_SUCCEEDED)
        self.assertIsNotNone(processed.legacy_session_id)
        self.assertEqual(PracticeSession.objects.count(), 1)
        self.assertGreater(PracticeSession.objects.first().score, 4.9)
        processed.refresh_from_db()
        self.assertEqual(processed.partial_transcript, "clear practice text")
        refresh.assert_called_once()

    def test_scoring_job_status_returns_partial_transcript(self):
        job = ScoringJob.objects.create(
            script_name="Status Drill",
            script_text="clear practice text",
            audio_path="recording.txt",
            provider="uploaded_transcript",
            status=ScoringJob.STATUS_RUNNING,
            partial_transcript="clear practice",
        )

        response = self.client.get(reverse("practice:scoring_job_status", args=[job.pk]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["is_pending"])
        self.assertEqual(payload["partial_transcript"], "clear practice")

    def test_scoring_job_status_returns_final_transcript_render(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Status Done Drill",
            script_text="clear practice text",
            audio_path="recording.webm",
            transcript="clear text",
            wer=0.3,
            cer=0.25,
            clarity=0.7,
            score=3.0,
            artic_rate=142,
            pause_ratio=0.12,
            avg_conf=0.91,
            segments=(
                '[{"text": "clear", "start": 0.1, "end": 0.8}, '
                '{"text": " text", "start": 0.8, "end": 1.4}]'
            ),
        )
        SessionError.objects.create(
            session_id=session.pk,
            timestamp=session.timestamp,
            script_name=session.script_name,
            hyp_token="text",
            op="sub",
            error_kind="char_insert",
            hyp_start=6,
            hyp_end=10,
        )
        job = ScoringJob.objects.create(
            script_name=session.script_name,
            script_text=session.script_text,
            audio_path=session.audio_path,
            provider="uploaded_transcript",
            status=ScoringJob.STATUS_SUCCEEDED,
            partial_transcript="clear text",
            legacy_session_id=session.pk,
        )

        response = self.client.get(reverse("practice:scoring_job_status", args=[job.pk]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["is_done"])
        self.assertEqual(payload["session_id"], session.pk)
        self.assertEqual(payload["metrics"]["score"], "3")
        self.assertEqual(
            payload["score_text"],
            "Score: 3.00/5 | WER: 30.00% | CER: 25.00% | "
            "Clarity: 70.00% | Rate: 142 wpm | Pauses: 12% | Conf: 91%",
        )
        self.assertIn('data-start="0.100"', payload["transcript_html"])
        self.assertIn("err-char-insert", payload["transcript_html"])

    def test_session_detail_edits_transcript_and_refreshes_errors(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Edit Drill",
            script_text="clear practice text",
            audio_path="recording.txt",
            transcript="clear text",
            wer=0.3,
            clarity=0.7,
            score=3.0,
        )

        response = self.client.post(
            reverse("practice:session_detail", args=[session.pk]),
            {"transcript": "clear practice text"},
        )

        self.assertEqual(response.status_code, 302)
        session.refresh_from_db()
        self.assertEqual(session.transcript, "clear practice text")
        self.assertEqual(session.wer, 0.0)
        self.assertEqual(SessionError.objects.filter(session_id=session.pk).count(), 0)

    def test_session_detail_saves_self_review_notes(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Free Speak",
            script_text="",
            audio_path="recording.txt",
            transcript="today I talked freely",
        )

        response = self.client.post(
            reverse("practice:session_detail", args=[session.pk]),
            {
                "action": "save_self_review",
                "self_review_notes": "I rushed the last sentence.",
            },
        )

        self.assertEqual(response.status_code, 302)
        session.refresh_from_db()
        self.assertEqual(session.self_review_notes, "I rushed the last sentence.")

    def test_session_detail_creates_cards_from_self_review_notes(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Free Speak",
            script_text="",
            audio_path="recording.txt",
            transcript="I asked just one question and then rushed the ending",
        )

        response = self.client.post(
            reverse("practice:session_detail", args=[session.pk]),
            {
                "action": "create_cards_from_self_review",
                "provider": "local_template",
                "self_review_notes": (
                    "I rushed the last sentence.\n"
                    "I swallowed final consonants on asked and just."
                ),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ImprovementCard.objects.count(), 2)
        targets = set(ImprovementCard.objects.values_list("target_key", flat=True))
        self.assertIn("rushing", targets)
        self.assertIn("final consonants", targets)
        card = ImprovementCard.objects.get(target_key="rushing")
        self.assertEqual(card.stats["source"], "self_review")
        self.assertEqual(card.stats["source_session_id"], session.pk)

    def test_session_detail_clears_transcript(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Clear Drill",
            script_text="clear practice text",
            audio_path="recording.txt",
            transcript="wrong text",
            wer=0.5,
            clarity=0.5,
            score=2.5,
        )
        SessionError.objects.create(
            session_id=session.pk,
            timestamp=session.timestamp,
            script_name=session.script_name,
            ref_token="practice",
            op="del",
            error_kind="word_missing",
        )

        response = self.client.post(
            reverse("practice:session_detail", args=[session.pk]),
            {"action": "clear_transcript"},
        )

        self.assertEqual(response.status_code, 302)
        session.refresh_from_db()
        self.assertEqual(session.transcript, "")
        self.assertIsNone(session.score)
        self.assertEqual(SessionError.objects.filter(session_id=session.pk).count(), 0)

    def test_session_delete_removes_audio_and_history_entry(self):
        self._create_legacy_tables()
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "delete-me.webm"
            audio_path.write_bytes(b"audio")
            session = PracticeSession.objects.create(
                timestamp="2026-06-14T12:00:00",
                script_name="Delete Drill",
                script_text="clear practice text",
                audio_path=str(audio_path),
                transcript="clear practice text",
            )
            SessionError.objects.create(
                session_id=session.pk,
                timestamp=session.timestamp,
                script_name=session.script_name,
                ref_token="practice",
                op="del",
                error_kind="word_missing",
            )
            card = ImprovementCard.objects.create(
                title="Practice clear",
                kind=ImprovementCard.KIND_WORD,
                target_key="clear",
            )
            PracticeReview.objects.create(card=card, legacy_session_id=session.pk)
            job = ScoringJob.objects.create(
                script_name=session.script_name,
                script_text=session.script_text,
                audio_path=str(audio_path),
                provider="uploaded_transcript",
                legacy_session_id=session.pk,
            )

            response = self.client.post(reverse("practice:session_delete", args=[session.pk]))

            self.assertRedirects(response, reverse("practice:sessions"))
            self.assertFalse(audio_path.exists())
            self.assertFalse(PracticeSession.objects.filter(pk=session.pk).exists())
            self.assertEqual(SessionError.objects.filter(session_id=session.pk).count(), 0)
            self.assertEqual(PracticeReview.objects.filter(legacy_session_id=session.pk).count(), 0)
            job.refresh_from_db()
            self.assertIsNone(job.legacy_session_id)

    def test_session_audio_supports_byte_ranges_for_seek(self):
        self._create_legacy_tables()
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.webm"
            audio_path.write_bytes(b"0123456789")
            session = PracticeSession.objects.create(
                timestamp="2026-06-14T12:00:00",
                script_name="Audio Drill",
                script_text="clear practice text",
                audio_path=str(audio_path),
                transcript="clear practice text",
            )

            response = self.client.get(
                reverse("practice:session_audio", args=[session.pk]),
                HTTP_RANGE="bytes=2-5",
            )

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.headers["Accept-Ranges"], "bytes")
        self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(response.content, b"2345")

    def test_session_detail_renders_timed_transcript_segments(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Timed Drill",
            script_text="clear practice text",
            audio_path="recording.txt",
            transcript="clear practice text",
            segments='[{"text": "clear practice", "start": 1.25, "end": 2.5}, {"text": "text", "start": 2.5, "end": 3.0}]',
        )

        response = self.client.get(reverse("practice:session_detail", args=[session.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-timed-transcript")
        self.assertContains(response, 'data-start="1.250"')

    def test_session_detail_timed_segments_survive_punctuation(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Punctuated Timed Drill",
            script_text="Hello world again",
            audio_path="recording.txt",
            transcript="Hello, world again.",
            segments='[{"text": "Hello world", "start": 0.5, "end": 1.25}, {"text": "again", "start": 1.25, "end": 1.8}]',
        )

        response = self.client.get(reverse("practice:session_detail", args=[session.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-start="0.500"')
        self.assertContains(response, 'data-end="1.250"')
        self.assertContains(response, 'class="timed-transcript-segment"', count=2)

    def test_session_detail_renders_copy_score_text(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Copy Score Drill",
            script_text="clear practice text",
            audio_path="recording.txt",
            transcript="clear text",
            wer=0.3,
            cer=0.25,
            clarity=0.7,
            score=3.0,
            artic_rate=142,
            pause_ratio=0.12,
            avg_conf=0.91,
        )

        response = self.client.get(reverse("practice:session_detail", args=[session.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Copy score")
        self.assertContains(response, "data-score-copy-text")
        self.assertContains(
            response,
            "Score: 3.00/5 | WER: 30.00% | CER: 25.00% | "
            "Clarity: 70.00% | Rate: 142 wpm | Pauses: 12% | Conf: 91%",
        )

    def test_session_report_exports_markdown(self):
        self._create_legacy_tables()
        session = PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Report Drill",
            script_text="clear practice text",
            audio_path="recording.txt",
            transcript="clear text",
            wer=0.3,
            clarity=0.7,
            score=3.0,
        )
        SessionError.objects.create(
            session_id=session.pk,
            timestamp=session.timestamp,
            script_name=session.script_name,
            ref_token="practice",
            op="del",
            error_kind="word_missing",
        )

        response = self.client.get(reverse("practice:session_report", args=[session.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "text/markdown; charset=utf-8")
        self.assertContains(response, "# SpeechPractice Report: Report Drill")
        self.assertContains(response, "word_missing")

    def test_generated_card_drill_updates_card_review_schedule(self):
        self._create_legacy_tables()
        card = ImprovementCard.objects.create(
            title="Word focus: clear",
            kind=ImprovementCard.KIND_WORD,
            target_key="clear",
            prompt="Practice clear in short phrases.",
            mastery=0.2,
        )
        script = PracticeScript.objects.create(
            title="Drill: clear",
            body="clear practice text",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
            source_ref=f"card:{card.pk}",
        )
        GeneratedPracticeScript.objects.create(
            card=card,
            script=script,
            model_provider="test",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "recording.txt"
            path.write_text("clear practice text", encoding="utf-8")
            job = create_scoring_job(
                script=script,
                audio_path=str(path),
                provider="uploaded_transcript",
            )

            with patch("practice.services.jobs.refresh_improvement_cards"):
                processed = process_scoring_job(job.pk)

        card.refresh_from_db()
        self.assertEqual(processed.card_id, card.pk)
        self.assertEqual(PracticeReview.objects.filter(card=card).count(), 1)
        self.assertGreater(card.mastery, 0.2)
        self.assertIsNotNone(card.last_reviewed_at)
        self.assertGreater(card.due_at, timezone.now())

    def test_practice_post_queues_scoring_job(self):
        script = PracticeScript.objects.create(
            title="Queue Drill",
            body="queue this text",
            source=PracticeScript.SOURCE_USER,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "upload.txt"
            path.write_text("queue this text", encoding="utf-8")
            with path.open("rb") as upload:
                with patch("practice.views.enqueue_scoring_job") as enqueue:
                    response = self.client.post(
                        reverse("practice:practice"),
                        {
                            "script": script.pk,
                            "provider": "uploaded_transcript",
                            "audio": upload,
                        },
                    )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ScoringJob.objects.count(), 1)
        enqueue.assert_called_once()

    def test_practice_post_ajax_returns_job_status_payload(self):
        script = PracticeScript.objects.create(
            title="Ajax Queue Drill",
            body="queue this text",
            source=PracticeScript.SOURCE_USER,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "upload.txt"
            path.write_text("queue this text", encoding="utf-8")
            with path.open("rb") as upload:
                with patch("practice.views.enqueue_scoring_job") as enqueue:
                    response = self.client.post(
                        reverse("practice:practice"),
                        {
                            "script": script.pk,
                            "provider": "uploaded_transcript",
                            "audio": upload,
                        },
                        HTTP_ACCEPT="application/json",
                        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
                    )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("/jobs/", payload["status_url"])
        self.assertEqual(ScoringJob.objects.count(), 1)
        enqueue.assert_called_once()

    def test_free_speak_post_queues_transcription_only_job(self):
        script = PracticeScript.objects.create(
            title="Unused Drill",
            body="this text is not required",
            source=PracticeScript.SOURCE_USER,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "free.txt"
            path.write_text("free speech transcript", encoding="utf-8")
            with path.open("rb") as upload:
                with patch("practice.views.enqueue_scoring_job"):
                    response = self.client.post(
                        reverse("practice:practice"),
                        {
                            "mode": "free_speak",
                            "script": script.pk,
                            "provider": "uploaded_transcript",
                            "audio": upload,
                        },
                    )

        self.assertEqual(response.status_code, 302)
        job = ScoringJob.objects.get()
        self.assertEqual(job.mode, ScoringJob.MODE_FREE)
        self.assertEqual(job.script_name, "Free Speak")
        self.assertEqual(job.script_text, "")

    def test_free_speak_job_saves_unscored_history_entry(self):
        self._create_legacy_tables()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "free.txt"
            path.write_text("free speech transcript", encoding="utf-8")
            job = ScoringJob.objects.create(
                script_name="Free Speak",
                script_text="",
                audio_path=str(path),
                provider="uploaded_transcript",
                mode=ScoringJob.MODE_FREE,
            )

            with patch("practice.services.jobs.refresh_improvement_cards") as refresh:
                processed = process_scoring_job(job.pk)

        self.assertEqual(processed.status, ScoringJob.STATUS_SUCCEEDED)
        session = PracticeSession.objects.get()
        self.assertEqual(session.script_name, "Free Speak")
        self.assertEqual(session.transcript, "free speech transcript")
        self.assertIsNone(session.score)
        refresh.assert_not_called()

    def test_practice_post_preserves_explicit_card_context(self):
        card = ImprovementCard.objects.create(
            title="Sound pattern: R",
            kind=ImprovementCard.KIND_SOUND,
            target_key="R",
            prompt="Practice R.",
            mastery=0.3,
        )
        script = PracticeScript.objects.create(
            title="Custom R Drill",
            body="round red words",
            source=PracticeScript.SOURCE_USER,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "upload.txt"
            path.write_text("round red words", encoding="utf-8")
            with path.open("rb") as upload:
                with patch("practice.views.enqueue_scoring_job"):
                    response = self.client.post(
                        reverse("practice:practice"),
                        {
                            "script": script.pk,
                            "card": card.pk,
                            "provider": "uploaded_transcript",
                            "audio": upload,
                        },
                    )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ScoringJob.objects.get().card, card)

    def test_card_detail_lists_generated_drills_and_reviews(self):
        card = ImprovementCard.objects.create(
            title="Sound pattern: TH",
            kind=ImprovementCard.KIND_SOUND,
            target_key="TH",
            prompt="Practice TH.",
            mastery=0.4,
        )
        script = PracticeScript.objects.create(
            title="Drill: TH",
            body="thin things thrive",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=script)
        PracticeReview.objects.create(card=card, score=4.0, error_rate=0.1)

        response = self.client.get(reverse("practice:card_detail", args=[card.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Drill: TH")
        self.assertContains(response, "Review history")
        self.assertContains(response, "Local template")

    def test_card_detail_shows_matching_mistake_snippets(self):
        self._create_legacy_tables()
        card = ImprovementCard.objects.create(
            title="Word focus: night",
            kind=ImprovementCard.KIND_WORD,
            target_key="night",
            prompt="Practice night.",
            mastery=0.0,
        )
        script_text = "Silent night arrives before the bright morning."
        session = PracticeSession.objects.create(
            timestamp="2026-06-17T10:00:00",
            script_name="Solstice Drill",
            script_text=script_text,
            audio_path="",
            transcript="silent light arrives before the bright morning",
            score=2.0,
        )
        start = script_text.lower().index("night")
        SessionError.objects.create(
            session_id=session.pk,
            timestamp=session.timestamp,
            script_name=session.script_name,
            ref_token="night",
            hyp_token="light",
            op="sub",
            error_kind="char_replace",
            ref_start=start,
            ref_end=start + len("night"),
            ref_local_start=0,
            ref_local_end=len("night"),
            ref_token_len=len("night"),
        )

        response = self.client.get(reverse("practice:card_detail", args=[card.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Solstice Drill")
        self.assertContains(response, "<mark>night</mark>", html=True)
        self.assertContains(response, "Expected night")
        self.assertContains(response, "heard light")

    def test_card_delete_removes_card_but_keeps_generated_script(self):
        card = ImprovementCard.objects.create(
            title="Word focus: clear",
            kind=ImprovementCard.KIND_WORD,
            target_key="clear",
            prompt="Practice clear.",
            mastery=0.3,
        )
        script = PracticeScript.objects.create(
            title="Drill: clear",
            body="clear words carry",
            practice_kind=PracticeScript.KIND_DRILL,
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=script)

        response = self.client.post(reverse("practice:card_delete", args=[card.pk]))

        self.assertRedirects(response, reverse("practice:cards"))
        self.assertFalse(ImprovementCard.objects.filter(pk=card.pk).exists())
        self.assertTrue(PracticeScript.objects.filter(pk=script.pk).exists())

    def test_card_list_surfaces_refresh_date_range_controls(self):
        ImprovementCard.objects.create(
            title="Word focus: clear",
            kind=ImprovementCard.KIND_WORD,
            target_key="clear",
            prompt="Practice clear.",
            mastery=0.3,
            stats={"source_window_label": "2026-06-01 to 2026-06-19"},
        )

        response = self.client.get(reverse("practice:cards"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Start date")
        self.assertContains(response, "End date")
        self.assertContains(response, 'type="date"', count=2)
        self.assertContains(response, "From 2026-06-01 to 2026-06-19")

    def test_card_refresh_uses_submitted_date_range(self):
        with patch("practice.views.refresh_improvement_cards", return_value=3) as refresh:
            response = self.client.post(
                reverse("practice:cards"),
                {
                    "refresh": "1",
                    "start": "2026-06-01",
                    "end": "2026-06-19",
                },
            )

        self.assertRedirects(response, reverse("practice:cards"))
        refresh.assert_called_once()
        kwargs = refresh.call_args.kwargs
        self.assertEqual(kwargs["start_dt"].date().isoformat(), "2026-06-01")
        self.assertEqual(kwargs["end_dt"].date().isoformat(), "2026-06-19")
        self.assertEqual(kwargs["end_dt"].hour, 23)

    def test_card_refresh_requires_date_range(self):
        with patch("practice.views.refresh_improvement_cards") as refresh:
            response = self.client.post(reverse("practice:cards"), {"refresh": "1", "start": "", "end": ""})

        self.assertRedirects(response, reverse("practice:cards"))
        refresh.assert_not_called()

    def test_generate_script_view_records_provider_metadata(self):
        card = ImprovementCard.objects.create(
            title="Word focus: steady",
            kind=ImprovementCard.KIND_WORD,
            target_key="steady",
            prompt="Practice steady.",
            mastery=0.3,
        )

        response = self.client.post(
            reverse("practice:generate_script", args=[card.pk]),
            {"provider": "local_template", "next": "practice"},
        )

        self.assertEqual(response.status_code, 302)
        record = GeneratedPracticeScript.objects.get(card=card)
        self.assertEqual(record.model_provider, "local_template")
        self.assertEqual(record.auth_source, "local")
        self.assertEqual(record.script.practice_kind, PracticeScript.KIND_DRILL)
        self.assertEqual(record.script.source_ref, f"card:{card.pk}")
        self.assertIn("mode=quick", response["Location"])
        self.assertIn(f"script={record.script.pk}", response["Location"])

    def test_account_page_saves_encrypted_settings_and_models(self):
        with patch("practice.views.clear_local_whisper_cache") as clear_cache:
            response = self.client.post(
                reverse("practice:account"),
                {
                    "transcription_provider": "openai",
                    "script_generation_provider": "anthropic",
                    "openai_script_model": "gpt-5.4-mini",
                    "anthropic_script_model": "claude-sonnet-4-6",
                    "openai_transcription_model": "whisper-1",
                    "whisper_model_name": "small.en",
                    "whisper_device": "cpu",
                    "whisper_preset": "fast_cpu",
                    "whisper_language": "en",
                    "whisper_timestamps": "on",
                    "whisper_beam_size": 1,
                    "whisper_temperature": 0.0,
                    "whisper_no_speech_threshold": 0.25,
                    "whisper_condition_on_previous_text": "on",
                    "whisper_chunk_seconds": 90,
                    "autumn_base_url": "https://autumn.example.test",
                    "autumn_project": "Speech Practice",
                    "autumn_subprojects_text": "Drills, Review",
                    "openai_api_key": "sk-test",
                    "anthropic_api_key": "ak-test",
                    "autumn_token": "autumn-test",
                },
            )

        self.assertEqual(response.status_code, 302)
        clear_cache.assert_called_once()
        settings_obj = PracticeSettings.load()
        self.assertEqual(settings_obj.transcription_provider, "openai")
        self.assertEqual(settings_obj.openai_script_model, "gpt-5.4-mini")
        self.assertEqual(settings_obj.anthropic_script_model, "claude-sonnet-4-6")
        self.assertEqual(settings_obj.whisper_model_name, "small.en")
        self.assertEqual(settings_obj.whisper_preset, "fast_cpu")
        self.assertEqual(settings_obj.whisper_chunk_seconds, 90)
        self.assertEqual(settings_obj.autumn_project, "Speech Practice")
        self.assertEqual(settings_obj.autumn_subprojects, ["Drills", "Review"])
        self.assertEqual(settings_obj.get_secret("openai_api_key"), "sk-test")
        self.assertEqual(settings_obj.get_secret("anthropic_api_key"), "ak-test")
        self.assertEqual(settings_obj.get_secret("autumn_token"), "autumn-test")

    def test_account_page_renders_current_model_choices(self):
        response = self.client.get(reverse("practice:account"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GPT-5.5")
        self.assertContains(response, "Claude Sonnet 4.6")
        self.assertContains(response, "Local Whisper tuning")
        self.assertContains(response, "Settings sections")
        self.assertContains(response, "Advanced Whisper parameters")
        self.assertContains(response, "Connection status summary")
        self.assertContains(response, "Autumn timer")
        self.assertContains(response, "Autumn username")

    def test_account_page_connects_autumn_with_username_and_password(self):
        fake_client = Mock()
        fake_client.base_url = "https://autumn.example.test"
        fake_client.authenticate.return_value = "autumn-login-token"
        fake_client.list_projects.return_value = ["Speech", "Writing"]
        fake_client.list_subprojects.return_value = ["drills", "reading"]

        with patch("practice.views.AutumnClient", return_value=fake_client) as client_cls:
            response = self.client.post(
                reverse("practice:account"),
                {
                    "connect_autumn": "1",
                    "autumn_base_url": "https://autumn.example.test/",
                    "autumn_username": "reader",
                    "autumn_password": "secret",
                },
            )

        self.assertEqual(response.status_code, 302)
        client_cls.assert_called_once_with("https://autumn.example.test")
        fake_client.authenticate.assert_called_once_with("reader", "secret")
        fake_client.list_projects.assert_called_once()
        fake_client.list_subprojects.assert_called_once_with("Speech")
        settings_obj = PracticeSettings.load()
        self.assertEqual(settings_obj.autumn_base_url, "https://autumn.example.test")
        self.assertEqual(settings_obj.autumn_project, "Speech")
        self.assertEqual(settings_obj.get_secret("autumn_token"), "autumn-login-token")

    def test_account_page_renders_autumn_project_and_subproject_selectors(self):
        settings_obj = PracticeSettings.load()
        settings_obj.autumn_base_url = "https://autumn.example.test"
        settings_obj.autumn_project = "Speech"
        settings_obj.autumn_subprojects = ["drills"]
        settings_obj.set_secret("autumn_token", "autumn-test")
        settings_obj.save()
        fake_client = Mock()
        fake_client.list_projects.return_value = ["Speech", "Writing"]
        fake_client.list_subprojects.return_value = ["drills", "misc", "reading"]

        with patch("practice.views._autumn_token_client", return_value=fake_client):
            response = self.client.get(reverse("practice:account"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<select name="autumn_project"', html=False)
        self.assertContains(response, '<option value="Speech" selected>Speech</option>', html=True)
        self.assertContains(response, 'name="autumn_subprojects"', count=3)
        self.assertContains(response, 'value="drills"', html=False)
        self.assertContains(response, 'id="id_autumn_subprojects_0" checked', html=False)
        fake_client.list_projects.assert_called_once()
        fake_client.list_subprojects.assert_called_once_with("Speech")

    def test_account_page_saves_selected_autumn_subprojects(self):
        settings_obj = PracticeSettings.load()
        settings_obj.autumn_base_url = "https://autumn.example.test"
        settings_obj.autumn_project = "Speech"
        settings_obj.set_secret("autumn_token", "autumn-test")
        settings_obj.save()
        fake_client = Mock()
        fake_client.list_projects.return_value = ["Speech", "Writing"]
        fake_client.list_subprojects.return_value = ["drills", "misc", "reading"]

        with patch("practice.views._autumn_token_client", return_value=fake_client):
            response = self.client.post(
                reverse("practice:account"),
                {
                    "transcription_provider": settings_obj.transcription_provider,
                    "script_generation_provider": settings_obj.script_generation_provider,
                    "openai_script_model": settings_obj.openai_script_model,
                    "anthropic_script_model": settings_obj.anthropic_script_model,
                    "openai_transcription_model": settings_obj.openai_transcription_model,
                    "whisper_model_name": settings_obj.whisper_model_name,
                    "whisper_device": settings_obj.whisper_device,
                    "whisper_preset": settings_obj.whisper_preset,
                    "whisper_language": settings_obj.whisper_language,
                    "whisper_timestamps": "on",
                    "whisper_beam_size": settings_obj.whisper_beam_size,
                    "whisper_temperature": settings_obj.whisper_temperature,
                    "whisper_no_speech_threshold": settings_obj.whisper_no_speech_threshold,
                    "whisper_condition_on_previous_text": "on",
                    "whisper_chunk_seconds": settings_obj.whisper_chunk_seconds,
                    "autumn_base_url": settings_obj.autumn_base_url,
                    "autumn_project": "Writing",
                    "autumn_subprojects": ["misc", "reading"],
                },
            )

        self.assertEqual(response.status_code, 302)
        settings_obj.refresh_from_db()
        self.assertEqual(settings_obj.autumn_project, "Writing")
        self.assertEqual(settings_obj.autumn_subprojects, ["misc", "reading"])

    def test_autumn_subprojects_endpoint_returns_project_choices(self):
        settings_obj = PracticeSettings.load()
        settings_obj.autumn_base_url = "https://autumn.example.test"
        settings_obj.autumn_project = "Speech"
        settings_obj.set_secret("autumn_token", "autumn-test")
        settings_obj.save()
        fake_client = Mock()
        fake_client.list_subprojects.return_value = ["drills", "reading"]

        with patch("practice.views._autumn_token_client", return_value=fake_client):
            response = self.client.get(
                reverse("practice:autumn_subprojects"),
                {"project": "Speech"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "subprojects": ["drills", "reading"]})
        fake_client.list_subprojects.assert_called_once_with("Speech")

    def test_account_page_clears_whisper_cache_on_request(self):
        with patch("practice.views.clear_local_whisper_cache") as clear_cache:
            response = self.client.post(
                reverse("practice:account"),
                {"clear_whisper_cache": "1"},
            )

        self.assertEqual(response.status_code, 302)
        clear_cache.assert_called_once()

    def test_account_page_starts_and_stops_autumn_timer(self):
        settings_obj = PracticeSettings.load()
        settings_obj.autumn_base_url = "https://autumn.example.test"
        settings_obj.autumn_project = "Speech Practice"
        settings_obj.autumn_subprojects = ["Drills"]
        settings_obj.set_secret("autumn_token", "autumn-test")
        settings_obj.save()
        fake_client = patch("practice.views._autumn_client").start()
        self.addCleanup(patch.stopall)
        fake_client.return_value.start_timer.return_value = {"session": {"id": 42}}
        fake_client.return_value.stop_timer.return_value = {"duration": 12.5}

        response = self.client.post(
            reverse("practice:account"),
            {"start_autumn_timer": "1", "autumn_note": "Starting practice"},
        )

        self.assertEqual(response.status_code, 302)
        settings_obj.refresh_from_db()
        self.assertEqual(settings_obj.autumn_active_session_id, 42)
        fake_client.return_value.start_timer.assert_called_once_with(
            "Speech Practice",
            ["Drills"],
            note="Starting practice",
        )

        response = self.client.post(
            reverse("practice:account"),
            {"stop_autumn_timer": "1", "autumn_note": "Finished practice"},
        )

        self.assertEqual(response.status_code, 302)
        settings_obj.refresh_from_db()
        self.assertIsNone(settings_obj.autumn_active_session_id)
        fake_client.return_value.stop_timer.assert_called_once_with(
            session_id=42,
            project="Speech Practice",
            note="Finished practice",
        )

    def test_practice_page_starts_autumn_timer_from_transport_controls(self):
        PracticeScript.objects.create(
            title="Autumn Reading",
            body="Start the timer before the practice take.",
            practice_kind=PracticeScript.KIND_READING,
            source=PracticeScript.SOURCE_USER,
        )
        settings_obj = PracticeSettings.load()
        settings_obj.autumn_base_url = "https://autumn.example.test"
        settings_obj.autumn_project = "Speech Practice"
        settings_obj.autumn_subprojects = ["Drills"]
        settings_obj.set_secret("autumn_token", "autumn-test")
        settings_obj.save()

        response = self.client.get(reverse("practice:practice"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Start Autumn")
        self.assertContains(response, "autumn-timer-form")
        self.assertContains(response, "data-autumn-toggle")

        with patch("practice.views._autumn_client") as fake_client:
            fake_client.return_value.start_timer.return_value = {"session": {"id": 84}}
            response = self.client.post(
                reverse("practice:autumn_timer"),
                {
                    "start_autumn_timer": "1",
                    "autumn_note": "SpeechPractice: Autumn Reading",
                    "next": reverse("practice:practice"),
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("practice:practice"))
        settings_obj.refresh_from_db()
        self.assertEqual(settings_obj.autumn_active_session_id, 84)
        fake_client.return_value.start_timer.assert_called_once_with(
            "Speech Practice",
            ["Drills"],
            note="SpeechPractice: Autumn Reading",
        )

    def test_practice_autumn_timer_endpoint_returns_json_without_redirect(self):
        settings_obj = PracticeSettings.load()
        settings_obj.autumn_base_url = "https://autumn.example.test"
        settings_obj.autumn_project = "Speech Practice"
        settings_obj.autumn_subprojects = ["Drills"]
        settings_obj.set_secret("autumn_token", "autumn-test")
        settings_obj.save()

        with patch("practice.views._autumn_client") as fake_client:
            fake_client.return_value.start_timer.return_value = {"session": {"id": 84}}
            response = self.client.post(
                reverse("practice:autumn_timer"),
                {
                    "start_autumn_timer": "1",
                    "autumn_note": "SpeechPractice: Autumn Reading",
                    "next": reverse("practice:practice"),
                },
                HTTP_ACCEPT="application/json",
                HTTP_X_REQUESTED_WITH="XMLHttpRequest",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["active"])
        self.assertEqual(payload["button_label"], "Stop Autumn")
        self.assertEqual(payload["button_name"], "stop_autumn_timer")
        self.assertNotIn("Location", response.headers)
        settings_obj.refresh_from_db()
        self.assertEqual(settings_obj.autumn_active_session_id, 84)

    def test_progress_page_renders_filters_and_chart_data(self):
        self._create_legacy_tables()
        PracticeSession.objects.create(
            timestamp="2026-06-14T12:00:00",
            script_name="Progress Drill",
            script_text="clear practice text",
            audio_path="recording.txt",
            transcript="clear practice text",
            wer=0.0,
            clarity=1.0,
            score=5.0,
        )

        response = self.client.get(reverse("practice:progress"), {"script": "Progress Drill"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Progress tracker")
        self.assertContains(response, 'data-progress-chart="score"')
        self.assertContains(response, "Progress Drill")

    def _create_legacy_tables(self):
        existing = set(connection.introspection.table_names())
        with connection.schema_editor() as schema:
            if PracticeSession._meta.db_table not in existing:
                schema.create_model(PracticeSession)
            if SessionError._meta.db_table not in existing:
                try:
                    schema.create_model(SessionError)
                except OperationalError:
                    pass
        SessionError.objects.all().delete()
        PracticeSession.objects.all().delete()
