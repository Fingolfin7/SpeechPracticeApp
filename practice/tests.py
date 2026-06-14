from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import db as legacy_db
from error_analytics import get_phrase_trend_summary
from django.db import connection
from django.db.utils import OperationalError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase
from django.urls import reverse
from django.utils import timezone

from .models import (
    GeneratedPracticeScript,
    ImprovementCard,
    PracticeReview,
    PracticeScript,
    PracticeSettings,
    PracticeSession,
    ScoringJob,
    SessionError,
)
from .services.jobs import create_scoring_job, process_scoring_job
from .services.scoring import score_transcript
from .services.script_import import import_script_items, parse_csv_text, parse_json_text
from .services.script_generation import generate_local_template, parse_generated_script, script_generation_provider_choices
from .services.analytics import build_card_candidates, today_queue
from .services.transcription import TranscriptResult, UploadedTranscriptProvider


class PracticeWebTests(TransactionTestCase):
    def test_script_library_page_renders(self):
        PracticeScript.objects.create(
            title="Breath Drill",
            body="A clean phrase begins with a calm breath.",
            source=PracticeScript.SOURCE_USER,
        )

        response = self.client.get(reverse("practice:scripts"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Breath Drill")
        self.assertContains(response, "Edit")
        self.assertContains(response, "Delete")

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
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=script)

        response = self.client.get(reverse("practice:dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Next reps")
        self.assertContains(response, "Sound pattern: S")
        self.assertContains(response, "Practice drill")

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
        self.assertIn("Focus area: Word focus: river", draft.prompt_snapshot)

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
        self.assertIn("steady breath today", draft.body)
        self.assertIn("inside a longer sentence", draft.body)

    def test_phrase_trend_summary_finds_error_context_phrase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "phrase-history.db")
            session = legacy_db.get_session(db_path)
            try:
                scored = legacy_db.add_session(
                    session,
                    script_name="Phrase Drill",
                    script_text="keep steady breath today",
                    audio_path="recording.txt",
                    transcript="keep steady today",
                    wer=0.25,
                    clarity=4.0,
                    score=3.5,
                )
                legacy_db.replace_session_errors(
                    session,
                    scored.id,
                    scored.timestamp,
                    scored.script_name,
                    [
                        {
                            "ref_token": "breath",
                            "hyp_token": None,
                            "op": "del",
                            "error_kind": "word_missing",
                            "ref_start": 12,
                            "ref_end": 18,
                            "ref_local_start": 0,
                            "ref_local_end": 6,
                            "ref_token_len": 6,
                        }
                    ],
                )

                summary = get_phrase_trend_summary(session, top_n=3, min_attempts=1)
            finally:
                bind = session.get_bind()
                session.close()
                bind.dispose()

        phrases = summary["top_trouble_phrases"]
        self.assertEqual(phrases[0]["phrase"], "steady breath today")
        self.assertEqual(phrases[0]["errors"], 1.0)

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

    def test_generated_script_parser_handles_structured_output(self):
        title, body = parse_generated_script(
            "TITLE: Crisp TH Drill\nSCRIPT:\nThin things thrive.\nBreathe through the phrase.",
            fallback_title="Fallback",
        )

        self.assertEqual(title, "Crisp TH Drill")
        self.assertIn("Thin things thrive", body)
        self.assertNotIn("TITLE:", body)

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
        self.assertContains(response, 'data-script-preview-base="/scripts/0/preview/"')
        self.assertContains(response, "data-script-title")
        self.assertContains(response, "Find a script")
        self.assertContains(response, "Next random")

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
            source=PracticeScript.SOURCE_GENERATED,
        )
        latest_script = PracticeScript.objects.create(
            title="Latest steady drill",
            body="steady speech stays ready",
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=older_script)
        GeneratedPracticeScript.objects.create(card=card, script=latest_script)

        response = self.client.get(reverse("practice:practice"), {"card": card.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Word focus: steady")
        self.assertContains(response, "Latest steady drill")
        self.assertContains(response, f'name="card" value="{card.pk}"')

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
        refresh.assert_called_once()

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
            source=PracticeScript.SOURCE_GENERATED,
        )
        GeneratedPracticeScript.objects.create(card=card, script=script)
        PracticeReview.objects.create(card=card, score=4.0, error_rate=0.1)

        response = self.client.get(reverse("practice:card_detail", args=[card.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Drill: TH")
        self.assertContains(response, "Review history")
        self.assertContains(response, "Local template")

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
            {"provider": "local_template"},
        )

        self.assertEqual(response.status_code, 302)
        record = GeneratedPracticeScript.objects.get(card=card)
        self.assertEqual(record.model_provider, "local_template")
        self.assertEqual(record.script.source_ref, f"card:{card.pk}")

    def test_account_page_saves_encrypted_settings_and_models(self):
        response = self.client.post(
            reverse("practice:account"),
            {
                "transcription_provider": "openai",
                "script_generation_provider": "anthropic",
                "openai_script_model": "gpt-5.4-mini",
                "anthropic_script_model": "claude-sonnet-4-6",
                "openai_transcription_model": "whisper-1",
                "autumn_base_url": "https://autumn.example.test",
                "openai_api_key": "sk-test",
                "anthropic_api_key": "ak-test",
                "autumn_token": "autumn-test",
            },
        )

        self.assertEqual(response.status_code, 302)
        settings_obj = PracticeSettings.load()
        self.assertEqual(settings_obj.transcription_provider, "openai")
        self.assertEqual(settings_obj.openai_script_model, "gpt-5.4-mini")
        self.assertEqual(settings_obj.anthropic_script_model, "claude-sonnet-4-6")
        self.assertEqual(settings_obj.get_secret("openai_api_key"), "sk-test")
        self.assertEqual(settings_obj.get_secret("anthropic_api_key"), "ak-test")
        self.assertEqual(settings_obj.get_secret("autumn_token"), "autumn-test")

    def test_account_page_renders_current_model_choices(self):
        response = self.client.get(reverse("practice:account"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GPT-5.5")
        self.assertContains(response, "Claude Sonnet 4.6")

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
