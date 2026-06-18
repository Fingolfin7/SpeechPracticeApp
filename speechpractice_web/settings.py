from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "dev-only-speechpractice-secret-key-change-before-deploy",
)
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "practice",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "speechpractice_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "practice.context_processors.static_version",
            ],
        },
    },
]

WSGI_APPLICATION = "speechpractice_web.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.getenv("SPEECHPRACTICE_DB", str(BASE_DIR / "sessions.db")),
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("DJANGO_TIME_ZONE", "Europe/Prague")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_VERSION_CACHE_TIMEOUT = {
    "debug": 0,
    "production": 3600,
}

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LEGACY_DB_PATH = Path(DATABASES["default"]["NAME"])
LEGACY_SCRIPTS_DIR = Path(os.getenv("SPEECHPRACTICE_SCRIPTS_DIR", BASE_DIR / "scripts"))

TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "local_whisper")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
OPENAI_TRANSCRIPTION_MODEL_CHOICES = [
    ("whisper-1", "Whisper-1 - cloud, timestamped"),
]

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.3"))
WHISPER_CONDITION_ON_PREVIOUS_TEXT = os.getenv(
    "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
    "1",
) == "1"

SCORING_JOBS_INLINE = os.getenv("SCORING_JOBS_INLINE", "0") == "1"
CARD_REFRESH_WINDOW_DAYS = int(os.getenv("CARD_REFRESH_WINDOW_DAYS", "3650"))

SCRIPT_GENERATION_PROVIDER = os.getenv("SCRIPT_GENERATION_PROVIDER", "local_template")
OPENAI_SCRIPT_MODEL = os.getenv("OPENAI_SCRIPT_MODEL", "gpt-5.5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_SCRIPT_MODEL = os.getenv("ANTHROPIC_SCRIPT_MODEL", "claude-sonnet-4-6")

OPENAI_SCRIPT_MODEL_CHOICES = [
    ("gpt-5.5", "GPT-5.5"),
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4-mini", "GPT-5.4 mini"),
    ("gpt-5.4-nano", "GPT-5.4 nano"),
]
ANTHROPIC_SCRIPT_MODEL_CHOICES = [
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("claude-haiku-4-5", "Claude Haiku 4.5"),
]
DEFAULT_AUTUMN_BASE_URL = os.getenv("AUTUMN_BASE_URL", "http://127.0.0.1:8000")
CODEX_CLIENT_ID = os.getenv("CODEX_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
CODEX_AUTH_ISSUER = os.getenv("CODEX_AUTH_ISSUER", "https://auth.openai.com")
CODEX_CHATGPT_BASE_URL = os.getenv(
    "CODEX_CHATGPT_BASE_URL",
    "https://chatgpt.com/backend-api/codex",
)
