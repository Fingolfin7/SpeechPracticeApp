from __future__ import annotations

import os
from pathlib import Path

import dj_database_url
from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent

DEV_SECRET_KEY = "dev-only-speechpractice-secret-key-change-before-deploy"
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY") or os.getenv("SECRET_KEY") or DEV_SECRET_KEY
DEBUG = (os.getenv("DJANGO_DEBUG") or os.getenv("DEBUG") or "1").lower() in {
    "1",
    "true",
    "yes",
}
if not DEBUG and SECRET_KEY == DEV_SECRET_KEY:
    raise ImproperlyConfigured("Set DJANGO_SECRET_KEY before running in production.")

_default_hosts = "localhost,127.0.0.1,testserver"
if os.getenv("RENDER_EXTERNAL_HOSTNAME"):
    _default_hosts += f",{os.environ['RENDER_EXTERNAL_HOSTNAME']}"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("DJANGO_ALLOWED_HOSTS", _default_hosts).split(",")
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
if not DEBUG:
    MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")

REQUIRE_LOGIN = os.getenv("DJANGO_REQUIRE_LOGIN", "1") == "1"
if REQUIRE_LOGIN:
    authentication_index = MIDDLEWARE.index(
        "django.contrib.auth.middleware.AuthenticationMiddleware"
    )
    MIDDLEWARE.insert(
        authentication_index + 1,
        "django.contrib.auth.middleware.LoginRequiredMiddleware",
    )

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

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL:
    database_config = dj_database_url.parse(
        DATABASE_URL,
        conn_max_age=int(os.getenv("DB_CONN_MAX_AGE", "60")),
        conn_health_checks=True,
        ssl_require=os.getenv("DB_SSL_REQUIRE", "1") == "1",
    )
    database_config["DISABLE_SERVER_SIDE_CURSORS"] = os.getenv(
        "DB_DISABLE_SERVER_SIDE_CURSORS",
        "1" if "-pooler." in DATABASE_URL else "0",
    ) == "1"
    DATABASES = {"default": database_config}
else:
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

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
USE_S3 = os.getenv("USE_S3", "0") == "1"
STORAGES = {
    "default": {
        "BACKEND": (
            "storages.backends.s3.S3Storage"
            if USE_S3
            else "django.core.files.storage.FileSystemStorage"
        ),
    },
    "staticfiles": {
        "BACKEND": (
            "django.contrib.staticfiles.storage.StaticFilesStorage"
            if DEBUG
            else "whitenoise.storage.CompressedManifestStaticFilesStorage"
        ),
    },
}
STATIC_VERSION_CACHE_TIMEOUT = {
    "debug": 0,
    "production": 3600,
}

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME", "")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "eu-north-1")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL", "") or None
AWS_S3_SIGNATURE_VERSION = "s3v4"
AWS_S3_FILE_OVERWRITE = False
AWS_DEFAULT_ACL = None
AWS_QUERYSTRING_AUTH = True
AWS_QUERYSTRING_EXPIRE = int(os.getenv("AWS_QUERYSTRING_EXPIRE", "900"))
AWS_LOCATION = os.getenv("AWS_LOCATION", "")
if USE_S3 and not AWS_STORAGE_BUCKET_NAME:
    raise ImproperlyConfigured("Set AWS_STORAGE_BUCKET_NAME when USE_S3=1.")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LEGACY_SCRIPTS_DIR = Path(os.getenv("SPEECHPRACTICE_SCRIPTS_DIR", BASE_DIR / "scripts"))

TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "openai")
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
SCORING_JOBS_MODE = os.getenv(
    "SCORING_JOBS_MODE",
    "thread" if DEBUG else "queue",
).strip().lower()
CARD_REFRESH_WINDOW_DAYS = int(os.getenv("CARD_REFRESH_WINDOW_DAYS", "3650"))

DATA_UPLOAD_MAX_MEMORY_SIZE = int(
    os.getenv("DATA_UPLOAD_MAX_MEMORY_SIZE", str(30 * 1024 * 1024))
)
FILE_UPLOAD_MAX_MEMORY_SIZE = int(
    os.getenv("FILE_UPLOAD_MAX_MEMORY_SIZE", str(2 * 1024 * 1024))
)
OPENAI_DIRECT_UPLOAD_MAX_BYTES = int(
    os.getenv("OPENAI_DIRECT_UPLOAD_MAX_BYTES", str(24 * 1024 * 1024))
)

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "practice:dashboard"
LOGOUT_REDIRECT_URL = "login"

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]
if os.getenv("RENDER_EXTERNAL_HOSTNAME"):
    CSRF_TRUSTED_ORIGINS.append(
        f"https://{os.environ['RENDER_EXTERNAL_HOSTNAME']}"
    )
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_SECURE_HSTS_SECONDS", "3600"))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = os.getenv(
        "DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", "0"
    ) == "1"
    SECURE_HSTS_PRELOAD = os.getenv("DJANGO_SECURE_HSTS_PRELOAD", "0") == "1"

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

# Test runs: swap the deliberately-slow PBKDF2 hasher for MD5. User creation and
# client.login() happen in nearly every test; with PBKDF2 that fixed cost was
# ~1s per test (~100s suite). Never affects real servers.
import sys  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "test":
    PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
