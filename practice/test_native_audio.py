import base64
import hashlib
import io
import json
import tempfile
import uuid
import wave
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from practice.models import (NativeAudioAssessment, NativeAudioTake, PracticeSession, SessionError,
                             ImprovementCard, PracticeLadder, LadderStepProgress, ScoringJob)
from practice.services import native_audio as service


def wav_bytes(seconds=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
        wav.writeframes(b"\0\0" * 8000 * seconds)
    return buffer.getvalue()


def response(content, refusal=None, finish_reason="stop"):
    usage = MagicMock(); usage.model_dump.return_value = {"total_tokens": 123}
    return SimpleNamespace(_request_id="req-1", model="gpt-audio-version", usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(refusal=refusal, content=content), finish_reason=finish_reason)])


VALID = {"status": "ok", "overall_intelligibility": 4, "articulation": 3, "pacing": 4,
         "observations": ["Ending unclear"], "strengths": ["Even pace"], "next_steps": ["Slow down"], "uncertainty": "Some noise"}


@override_settings(OPENAI_API_KEY="test-key")
class NativeAudioTests(TestCase):
    def setUp(self):
        self.media = tempfile.TemporaryDirectory(); self.override = override_settings(MEDIA_ROOT=self.media.name)
        self.override.enable(); self.addCleanup(self.media.cleanup); self.addCleanup(self.override.disable)
        self.owner = get_user_model().objects.create_user(username="native-owner", password="pass")
        self.other = get_user_model().objects.create_user(username="native-other", password="pass")
        self.client.force_login(self.owner)
        path = default_storage.save("native-audio/test.wav", ContentFile(wav_bytes()))
        self.take = NativeAudioTake.objects.create(user=self.owner, original_path=path, content_type="audio/wav", reference="Example phrase")

    def url(self, name):
        return reverse("practice:native_audio_" + name, args=[self.take.pk])

    def assessment(self):
        return NativeAudioAssessment.objects.create(take=self.take, submission_id=uuid.uuid4(),
            prompt_version=service.PROMPT_VERSION, prompt_text=service.PROMPT,
            prompt_sha256="a" * 64, requested_model=service.MODEL, parameters=service.PARAMETERS)

    @override_settings(REQUIRE_LOGIN=False)
    def test_owner_isolation_and_authentication(self):
        self.client.force_login(self.other)
        for name in ("take", "play"):
            self.assertEqual(self.client.get(self.url(name)).status_code, 404)
        for name in ("assess", "delete", "recover"):
            self.assertEqual(self.client.post(self.url(name), {"submission_id": uuid.uuid4()}).status_code, 404)
        self.client.logout(); self.assertEqual(self.client.get(self.url("take")).status_code, 302)

    @patch("openai.OpenAI")
    def test_provider_contract_audit_duplicate_deletion_and_legacy_isolation(self, openai):
        openai.return_value.chat.completions.create.return_value = response(json.dumps(VALID))
        legacy = (PracticeSession, SessionError, ImprovementCard, PracticeLadder, LadderStepProgress, ScoringJob)
        counts_before = {model: model.objects.count() for model in legacy}
        submission = uuid.uuid4(); self.client.post(self.url("assess"), {"submission_id": submission})
        a = self.take.assessments.get(); self.take.refresh_from_db()
        self.assertEqual(a.status, "complete"); self.assertEqual(a.result, VALID)
        self.assertEqual((a.request_id, a.resolved_model, a.usage), ("req-1", "gpt-audio-version", {"total_tokens": 123}))
        self.assertEqual(a.prompt_sha256, hashlib.sha256(a.prompt_text.encode()).hexdigest())
        self.assertIn("Example phrase", a.prompt_text)
        self.assertEqual(a.sent_audio_sha256, self.take.wav_sha256)
        openai.assert_called_once_with(api_key="test-key", max_retries=0, timeout=90.0)
        kwargs = openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-audio-1.5"); self.assertEqual(kwargs["modalities"], ["text"]); self.assertFalse(kwargs["store"])
        data = base64.b64decode(kwargs["messages"][0]["content"][1]["input_audio"]["data"])
        with default_storage.open(self.take.wav_path, "rb") as wav:
            self.assertEqual(data, wav.read())
        for model in legacy:
            self.assertEqual(model.objects.count(), counts_before[model])
        self.client.post(self.url("assess"), {"submission_id": submission}); openai.return_value.chat.completions.create.assert_called_once()
        paths = [self.take.original_path, self.take.wav_path]
        self.assertTrue(all(default_storage.exists(p) for p in paths)); self.client.post(self.url("delete"))
        self.assertFalse(any(default_storage.exists(p) for p in paths)); self.assertFalse(NativeAudioAssessment.objects.exists())

    @patch("openai.OpenAI")
    def test_provider_failure_keeps_recording_retry_is_new(self, openai):
        openai.return_value.chat.completions.create.side_effect = RuntimeError("secret")
        self.client.post(self.url("assess"), {"submission_id": uuid.uuid4()})
        a = self.take.assessments.get(); self.assertEqual(a.status, "failed"); self.assertNotIn("secret", a.error); self.assertEqual(a.result, {})
        self.assertTrue(default_storage.exists(self.take.original_path))
        self.client.post(self.url("assess"), {"submission_id": uuid.uuid4()}); self.assertEqual(self.take.assessments.count(), 2)

    @patch("openai.OpenAI")
    def test_refusal_abstention_unstructured_and_truncation_have_no_ratings(self, openai):
        cases = [(response("", refusal="no"), "unavailable"), (response('{"status":"abstain","reason":"unclear"}'), "abstained"),
                 (response("not JSON"), "unstructured"), (response(json.dumps(VALID), finish_reason="length"), "unavailable")]
        for provider_response, expected in cases:
            openai.return_value.chat.completions.create.return_value = provider_response
            self.client.post(self.url("assess"), {"submission_id": uuid.uuid4()})
            a = self.take.assessments.first(); self.assertEqual(a.status, expected); self.assertEqual(a.result, {})

    @patch("practice.services.native_audio.run_assessment")
    def test_duplicate_running_and_recovery_do_not_call_provider(self, run):
        a = self.assessment(); self.client.post(self.url("assess"), {"submission_id": uuid.uuid4()}); run.assert_not_called()
        self.client.post(self.url("delete")); self.assertTrue(NativeAudioTake.objects.exists())
        NativeAudioAssessment.objects.filter(pk=a.pk).update(created_at=timezone.now() - timedelta(minutes=6))
        self.client.post(self.url("recover")); a.refresh_from_db(); self.assertEqual(a.status, "interrupted"); run.assert_not_called()

    def test_invalid_audio_and_upload_limit(self):
        too_large = SimpleUploadedFile("x.wav", b"x" * (service.MAX_UPLOAD + 1), content_type="audio/wav")
        self.client.post(reverse("practice:native_audio_create"), {"audio": too_large}); self.assertEqual(NativeAudioTake.objects.count(), 1)
        bad = SimpleUploadedFile("x.wav", b"not audio", content_type="audio/wav")
        self.client.post(reverse("practice:native_audio_create"), {"audio": bad})
        take = NativeAudioTake.objects.exclude(pk=self.take.pk).get()
        self.client.post(reverse("practice:native_audio_assess", args=[take.pk]), {"submission_id": uuid.uuid4()})
        self.assertEqual(take.assessments.get().status, "failed"); self.assertTrue(default_storage.exists(take.original_path))

    @patch("openai.OpenAI")
    def test_interrupted_during_conversion_does_not_start_provider_request(self, openai):
        assessment = self.assessment()
        def interrupt(_take):
            NativeAudioAssessment.objects.filter(pk=assessment.pk).update(status="interrupted")
            return b"converted"
        with patch("practice.services.native_audio.convert", side_effect=interrupt):
            service.run_assessment(assessment)
        openai.assert_not_called()
        assessment.refresh_from_db(); self.assertEqual(assessment.status, "interrupted")

    def test_conversion_rejects_audio_longer_than_two_minutes(self):
        path = default_storage.save("native-audio/long.wav", ContentFile(wav_bytes(121)))
        self.take.original_path = path; self.take.save(update_fields=["original_path"])
        with self.assertRaises(service.AudioError):
            service.convert(self.take)
        self.assertTrue(default_storage.exists(path))
    def test_validation_rejects_bad_ratings_and_malformed_report(self):
        for raw in ('{"status":"ok","overall_intelligibility":5}', '{"status":"abstain"}', 'null', json.dumps({**VALID, "pacing": True}), json.dumps({**VALID, "pacing": 6})):
            self.assertIsNone(service.validate(raw))
