import base64
import io
import json
import os
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, override_settings

from . import service, views


def audio(seconds=1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
        wav.writeframes(b"\0\0" * 8000 * seconds)
    return stream.getvalue()


RESULT = {"status":"ok", "summary":"Mostly clear", "strengths":["Even pace"], "observations":["An ending sounded soft"], "practice":["Slow the ending"], "uncertainty":"One phrase was quiet"}


def upload(seconds=1):
    return SimpleUploadedFile("recording.wav", audio(seconds), content_type="audio/wav")


@override_settings(ALLOWED_HOSTS=["testserver", "127.0.0.1", "localhost"])
class StudioTests(SimpleTestCase):
    def test_one_page_no_login_or_history(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "OpenAI API key")
        self.assertContains(page, "Send for feedback")
        self.assertNotContains(page, "Sign in")
        self.assertNotContains(page, "Saved recordings")
        self.assertEqual(self.client.get("/accounts/login/").status_code, 404)

    @patch("openai.OpenAI")
    def test_audio_sent_directly_as_wav_and_no_key_in_response(self, openai):
        msg = SimpleNamespace(content=json.dumps(RESULT), refusal=None)
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])
        response = self.client.post("/evaluate/", {"api_key":"private-test-key", "audio":upload()})
        self.assertEqual(response.status_code, 200); self.assertEqual(response.json(), RESULT)
        self.assertNotIn("private-test-key", response.content.decode())
        openai.assert_called_once_with(api_key="private-test-key", max_retries=0, timeout=90.0)
        args = openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(args["model"], "gpt-audio-1.5"); self.assertEqual(args["modalities"], ["text"]); self.assertFalse(args["store"])
        part = args["messages"][0]["content"][1]
        self.assertEqual(part["input_audio"]["format"], "wav")
        with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"])), "rb") as wav:
            self.assertEqual(wav.getframerate(), 24000)

    @patch("openai.OpenAI")
    def test_refusal_and_failure_never_expose_key(self, openai):
        msg = SimpleNamespace(content="", refusal="Cannot assess")
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])
        result = self.client.post("/evaluate/", {"api_key":"secret-key", "audio":upload()}).json()
        self.assertEqual(result["status"], "unavailable")
        openai.return_value.chat.completions.create.side_effect = RuntimeError("secret-key")
        result = self.client.post("/evaluate/", {"api_key":"secret-key", "audio":upload()})
        self.assertEqual(result.status_code, 502); self.assertNotIn("secret-key", result.content.decode())

    @patch("openai.OpenAI")
    def test_invalid_and_too_long_audio_do_not_call_provider(self, openai):
        result = self.client.post("/evaluate/", {"api_key":"secret", "audio":upload(121)})
        self.assertEqual(result.status_code, 400); self.assertIn("two minutes", result.json()["reason"])
        result = self.client.post("/evaluate/", {"api_key":"secret", "audio":SimpleUploadedFile("a", b"bad")})
        self.assertEqual(result.status_code, 400); openai.assert_not_called()

    def test_missing_key_and_simultaneous_request(self):
        self.assertEqual(self.client.post("/evaluate/", {"audio":upload()}).status_code, 400)
        views._active_request.acquire()
        try:
            self.assertEqual(self.client.post("/evaluate/", {"api_key":"secret", "audio":upload()}).status_code, 409)
        finally:
            views._active_request.release()

    def test_csrf_and_nonlocal_requests_rejected(self):
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.post("/evaluate/", {"api_key":"secret", "audio":upload()}).status_code, 403)
        page = client.get("/"); token = page.cookies["csrftoken"].value
        self.assertEqual(client.post("/evaluate/", {"csrfmiddlewaretoken":token, "api_key":"secret", "audio":upload()}, HTTP_ORIGIN="https://evil.example").status_code, 403)
        self.assertEqual(client.get("/", REMOTE_ADDR="192.0.2.5").status_code, 403)
        self.assertEqual(client.get("/", HTTP_HOST="evil.example").status_code, 400)

    def test_temporary_files_removed(self):
        with tempfile.TemporaryDirectory() as parent:
            with patch("tempfile.tempdir", parent):
                service.to_wav(upload())
                self.assertEqual(os.listdir(parent), [])
                with self.assertRaises(service.AudioError):
                    service.to_wav(SimpleUploadedFile("bad", b"bad"))
                self.assertEqual(os.listdir(parent), [])

    def test_response_validation(self):
        self.assertEqual(service.parse_response("```json\n" + json.dumps(RESULT) + "\n```"), RESULT)
        self.assertIsNone(service.parse_response('{"status":"ok","summary":"x"}'))
        self.assertEqual(service.parse_response('{"status":"abstain","reason":"unclear"}')["status"], "abstain")
