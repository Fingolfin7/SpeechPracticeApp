from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet
from django.conf import settings
from django.db import models
from django.utils import timezone


class PracticeSession(models.Model):
    timestamp = models.CharField(max_length=64)
    script_name = models.CharField(max_length=255)
    script_text = models.TextField()
    audio_path = models.TextField()
    transcript = models.TextField(blank=True, null=True)
    wer = models.FloatField(blank=True, null=True)
    clarity = models.FloatField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    segments = models.TextField(blank=True, null=True)
    cer = models.FloatField(blank=True, null=True)
    artic_rate = models.FloatField(blank=True, null=True)
    pause_ratio = models.FloatField(blank=True, null=True)
    filled_pauses = models.FloatField(blank=True, null=True)
    avg_conf = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = "sessions"
        ordering = ["-timestamp", "-id"]

    def __str__(self) -> str:
        return f"{self.timestamp} - {self.script_name}"


class SessionError(models.Model):
    session_id = models.IntegerField(db_index=True)
    timestamp = models.CharField(max_length=64, db_index=True)
    script_name = models.CharField(max_length=255, blank=True, null=True)
    ref_token = models.CharField(max_length=255, blank=True, null=True, db_index=True)
    hyp_token = models.CharField(max_length=255, blank=True, null=True)
    op = models.CharField(max_length=32, db_index=True)
    error_kind = models.CharField(max_length=64, db_index=True)
    ref_start = models.IntegerField(blank=True, null=True)
    ref_end = models.IntegerField(blank=True, null=True)
    hyp_start = models.IntegerField(blank=True, null=True)
    hyp_end = models.IntegerField(blank=True, null=True)
    ref_local_start = models.IntegerField(blank=True, null=True)
    ref_local_end = models.IntegerField(blank=True, null=True)
    hyp_local_start = models.IntegerField(blank=True, null=True)
    hyp_local_end = models.IntegerField(blank=True, null=True)
    ref_token_len = models.IntegerField(blank=True, null=True)
    hyp_token_len = models.IntegerField(blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    segment_start = models.FloatField(blank=True, null=True)
    segment_end = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = "session_errors"


class PracticeScript(models.Model):
    SOURCE_BUILTIN = "builtin"
    SOURCE_USER = "user"
    SOURCE_IMPORTED = "imported"
    SOURCE_GENERATED = "generated"
    SOURCE_CHOICES = [
        (SOURCE_BUILTIN, "Built-in"),
        (SOURCE_USER, "User"),
        (SOURCE_IMPORTED, "Imported"),
        (SOURCE_GENERATED, "Generated"),
    ]

    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255, blank=True)
    body = models.TextField()
    source = models.CharField(max_length=32, choices=SOURCE_CHOICES, default=SOURCE_USER)
    source_ref = models.CharField(max_length=512, blank=True)
    tags = models.JSONField(default=list, blank=True)
    target_patterns = models.JSONField(default=list, blank=True)
    difficulty = models.PositiveSmallIntegerField(default=1)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["title"]
        indexes = [
            models.Index(fields=["source", "active"]),
            models.Index(fields=["difficulty"]),
        ]

    def __str__(self) -> str:
        return self.title

    @property
    def word_count(self) -> int:
        return len(self.body.split())


class ImprovementCard(models.Model):
    KIND_WORD = "word"
    KIND_SOUND = "sound"
    KIND_CHARACTER = "character"
    KIND_POSITION = "position"
    KIND_PHRASE = "phrase"
    KIND_FLUENCY = "fluency"
    KIND_CHOICES = [
        (KIND_WORD, "Word"),
        (KIND_SOUND, "Sound"),
        (KIND_CHARACTER, "Character"),
        (KIND_POSITION, "Word Position"),
        (KIND_PHRASE, "Phrase"),
        (KIND_FLUENCY, "Fluency"),
    ]

    STATUS_LEARNING = "learning"
    STATUS_REVIEW = "review"
    STATUS_MASTERED = "mastered"
    STATUS_PAUSED = "paused"
    STATUS_CHOICES = [
        (STATUS_LEARNING, "Learning"),
        (STATUS_REVIEW, "Review"),
        (STATUS_MASTERED, "Mastered"),
        (STATUS_PAUSED, "Paused"),
    ]

    title = models.CharField(max_length=255)
    kind = models.CharField(max_length=32, choices=KIND_CHOICES)
    target_key = models.CharField(max_length=255)
    prompt = models.TextField(blank=True)
    stats = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_LEARNING)
    mastery = models.FloatField(default=0.0)
    due_at = models.DateTimeField(default=timezone.now)
    last_reviewed_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["due_at", "-updated_at"]
        constraints = [
            models.UniqueConstraint(fields=["kind", "target_key"], name="unique_card_target")
        ]
        indexes = [
            models.Index(fields=["kind", "status"]),
            models.Index(fields=["due_at"]),
        ]

    def __str__(self) -> str:
        return self.title


class PracticeReview(models.Model):
    card = models.ForeignKey(ImprovementCard, on_delete=models.CASCADE, related_name="reviews")
    legacy_session_id = models.IntegerField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    error_rate = models.FloatField(blank=True, null=True)
    notes = models.TextField(blank=True)
    reviewed_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-reviewed_at"]


class ScoringJob(models.Model):
    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_SUCCEEDED = "succeeded"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_QUEUED, "Queued"),
        (STATUS_RUNNING, "Running"),
        (STATUS_SUCCEEDED, "Succeeded"),
        (STATUS_FAILED, "Failed"),
    ]

    script = models.ForeignKey(
        PracticeScript,
        on_delete=models.SET_NULL,
        related_name="scoring_jobs",
        blank=True,
        null=True,
    )
    card = models.ForeignKey(
        ImprovementCard,
        on_delete=models.SET_NULL,
        related_name="scoring_jobs",
        blank=True,
        null=True,
    )
    script_name = models.CharField(max_length=255)
    script_text = models.TextField()
    audio_path = models.TextField()
    provider = models.CharField(max_length=64)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_QUEUED)
    legacy_session_id = models.IntegerField(blank=True, null=True, db_index=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(blank=True, null=True)
    finished_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "created_at"]),
            models.Index(fields=["provider"]),
        ]

    def __str__(self) -> str:
        return f"{self.script_name} ({self.status})"


class GeneratedPracticeScript(models.Model):
    card = models.ForeignKey(
        ImprovementCard,
        on_delete=models.SET_NULL,
        related_name="generated_scripts",
        blank=True,
        null=True,
    )
    script = models.ForeignKey(
        PracticeScript,
        on_delete=models.CASCADE,
        related_name="generation_records",
        blank=True,
        null=True,
    )
    model_provider = models.CharField(max_length=64, default="manual")
    prompt_snapshot = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]


_FERNET = None


def _fernet() -> Fernet:
    global _FERNET
    if _FERNET is None:
        digest = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
        _FERNET = Fernet(base64.urlsafe_b64encode(digest))
    return _FERNET


class PracticeSettings(models.Model):
    TRANSCRIPTION_LOCAL = "local_whisper"
    TRANSCRIPTION_OPENAI = "openai"
    TRANSCRIPTION_UPLOAD = "uploaded_transcript"
    TRANSCRIPTION_CHOICES = [
        (TRANSCRIPTION_LOCAL, "Local Whisper"),
        (TRANSCRIPTION_OPENAI, "OpenAI transcription"),
        (TRANSCRIPTION_UPLOAD, "Transcript sidecar"),
    ]

    script_generation_provider = models.CharField(
        max_length=32,
        default="local_template",
        choices=[
            ("local_template", "Local template"),
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
        ],
    )
    transcription_provider = models.CharField(
        max_length=64,
        default=TRANSCRIPTION_LOCAL,
        choices=TRANSCRIPTION_CHOICES,
    )
    openai_script_model = models.CharField(max_length=96, default="gpt-5.5")
    anthropic_script_model = models.CharField(max_length=96, default="claude-sonnet-4-6")
    openai_transcription_model = models.CharField(max_length=96, default="whisper-1")
    autumn_base_url = models.URLField(blank=True, default="")
    autumn_token_enc = models.BinaryField(null=True, blank=True, editable=False)
    openai_api_key_enc = models.BinaryField(null=True, blank=True, editable=False)
    anthropic_api_key_enc = models.BinaryField(null=True, blank=True, editable=False)
    codex_token_bundle_enc = models.BinaryField(null=True, blank=True, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "practice settings"

    def __str__(self) -> str:
        return "SpeechPractice settings"

    @classmethod
    def load(cls) -> "PracticeSettings":
        obj, _created = cls.objects.get_or_create(pk=1)
        return obj

    def set_secret(self, field: str, value: str | None) -> None:
        allowed = {
            "autumn_token": "autumn_token_enc",
            "openai_api_key": "openai_api_key_enc",
            "anthropic_api_key": "anthropic_api_key_enc",
            "codex_token_bundle": "codex_token_bundle_enc",
        }
        attr = allowed[field]
        if not value:
            setattr(self, attr, None)
            return
        setattr(self, attr, _fernet().encrypt(value.encode("utf-8")))

    def get_secret(self, field: str) -> str | None:
        allowed = {
            "autumn_token": "autumn_token_enc",
            "openai_api_key": "openai_api_key_enc",
            "anthropic_api_key": "anthropic_api_key_enc",
            "codex_token_bundle": "codex_token_bundle_enc",
        }
        data = getattr(self, allowed[field])
        if not data:
            return None
        try:
            return _fernet().decrypt(bytes(data)).decode("utf-8")
        except Exception:
            return None

    def has_secret(self, field: str) -> bool:
        allowed = {
            "autumn_token": "autumn_token_enc",
            "openai_api_key": "openai_api_key_enc",
            "anthropic_api_key": "anthropic_api_key_enc",
            "codex_token_bundle": "codex_token_bundle_enc",
        }
        return bool(getattr(self, allowed[field]))
