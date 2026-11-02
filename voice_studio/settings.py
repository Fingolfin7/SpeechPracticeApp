import secrets

SECRET_KEY = secrets.token_urlsafe(48)
DEBUG = False
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
ROOT_URLCONF = "voice_studio.urls"
INSTALLED_APPS = ["django.contrib.staticfiles", "voice_studio"]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "voice_studio.middleware.LoopbackOnly",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
TEMPLATES = [{"BACKEND": "django.template.backends.django.DjangoTemplates", "APP_DIRS": True,
              "OPTIONS": {"context_processors": ["django.template.context_processors.request"]}}]
STATIC_URL = "/static/"
FILE_UPLOAD_MAX_MEMORY_SIZE = 11 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 11 * 1024 * 1024
CSRF_COOKIE_SAMESITE = "Strict"
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
