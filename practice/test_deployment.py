from __future__ import annotations

import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, modify_settings, override_settings
from django.urls import reverse
from django.utils import timezone

from practice.models import PracticeScript, ScoringJob, PracticeSession
from practice.services.audio_storage import (
    audio_exists,
    delete_audio,
    materialized_audio,
    save_uploaded_audio,
)
from practice.services.jobs import enqueue_scoring_job, recover_stale_scoring_jobs


class DeploymentBehaviorTests(TestCase):
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
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.storage_override = override_settings(
            MEDIA_ROOT=self.temp_dir.name,
            STORAGES={
                "default": {
                    "BACKEND": "django.core.files.storage.FileSystemStorage",
                },
                "staticfiles": {
                    "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
                },
            },
        )
        self.storage_override.enable()
        self.addCleanup(self.storage_override.disable)

    def test_uploaded_audio_uses_storage_and_can_be_materialized(self):
        upload = SimpleUploadedFile("take.webm", b"webm-audio", content_type="audio/webm")

        audio_ref = save_uploaded_audio(upload, "Clear Speech")

        self.assertTrue(audio_ref.startswith("recordings/web/"))
        self.assertTrue(audio_exists(audio_ref))
        with materialized_audio(audio_ref) as local_path:
            self.assertEqual(Path(local_path).read_bytes(), b"webm-audio")
        self.assertTrue(delete_audio(audio_ref))
        self.assertFalse(audio_exists(audio_ref))

    def test_storage_backed_audio_supports_byte_ranges(self):
        audio_ref = default_storage.save(
            "recordings/web/range.webm",
            ContentFile(b"0123456789"),
        )
        session = PracticeSession.objects.create(
            timestamp="2026-06-24T12:00:00",
            script_name="Range",
            script_text="Range",
            audio_path=audio_ref,
        )

        response = self.client.get(
            reverse("practice:session_audio", args=[session.pk]),
            HTTP_RANGE="bytes=2-5",
        )

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, b"2345")
        self.assertEqual(response["Content-Range"], "bytes 2-5/10")
        self.assertEqual(response["Cache-Control"], "private, max-age=3600")

    def test_duplicate_submission_key_returns_existing_job(self):
        script = PracticeScript.objects.create(title="Retry", body="Retry safely.")
        submission_id = str(uuid.uuid4())

        with patch("practice.views.enqueue_scoring_job"):
            first = self.client.post(
                reverse("practice:practice"),
                {
                    "mode": "script",
                    "script": script.pk,
                    "provider": "openai",
                    "audio": SimpleUploadedFile("take.webm", b"first"),
                },
                HTTP_ACCEPT="application/json",
                HTTP_X_IDEMPOTENCY_KEY=submission_id,
            )
            second = self.client.post(
                reverse("practice:practice"),
                {
                    "mode": "script",
                    "script": script.pk,
                    "provider": "openai",
                    "audio": SimpleUploadedFile("take.webm", b"second"),
                },
                HTTP_ACCEPT="application/json",
                HTTP_X_IDEMPOTENCY_KEY=submission_id,
            )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(ScoringJob.objects.count(), 1)
        self.assertEqual(first.json()["id"], second.json()["id"])

    @override_settings(OPENAI_DIRECT_UPLOAD_MAX_BYTES=4)
    def test_oversized_audio_is_rejected_before_storage(self):
        script = PracticeScript.objects.create(title="Small upload", body="Keep it small.")

        response = self.client.post(
            reverse("practice:practice"),
            {
                "mode": "script",
                "script": script.pk,
                "provider": "openai",
                "audio": SimpleUploadedFile("large.webm", b"12345"),
            },
            HTTP_ACCEPT="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ScoringJob.objects.count(), 0)
        self.assertIn("reliable upload", response.json()["error"])

    @override_settings(SCORING_JOBS_INLINE=False, SCORING_JOBS_MODE="queue")
    def test_queue_mode_does_not_start_request_thread(self):
        job = ScoringJob.objects.create(
            script_name="Queued",
            script_text="Queued",
            audio_path="recordings/web/queued.webm",
            provider="openai",
        )
        with patch("practice.services.jobs.threading.Thread") as thread:
            enqueue_scoring_job(job)
        thread.assert_not_called()

    def test_stale_running_jobs_are_requeued(self):
        job = ScoringJob.objects.create(
            script_name="Interrupted",
            script_text="Interrupted",
            audio_path="recordings/web/interrupted.webm",
            provider="openai",
            status=ScoringJob.STATUS_RUNNING,
            started_at=timezone.now() - timedelta(hours=1),
        )

        recovered = recover_stale_scoring_jobs(stale_after_minutes=30)

        job.refresh_from_db()
        self.assertEqual(recovered, 1)
        self.assertEqual(job.status, ScoringJob.STATUS_QUEUED)
        self.assertIsNone(job.started_at)

    @modify_settings(
        MIDDLEWARE={
            "append": "django.contrib.auth.middleware.LoginRequiredMiddleware",
        }
    )
    def test_production_login_gate_exempts_health_check(self):
        self.client.logout()
        protected = self.client.get(reverse("practice:dashboard"))
        health = self.client.get(reverse("practice:health"))

        self.assertEqual(protected.status_code, 302)
        self.assertIn(reverse("login"), protected.url)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"ok": True})
