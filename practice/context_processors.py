from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.contrib.staticfiles import finders
from django.core.cache import cache


STATIC_VERSION_FILES = {
    "account_js": "practice/account.js",
    "app": "practice/app.css",
    "favicon": "practice/favicon.svg",
    "history_js": "practice/history.js",
    "job_status_js": "practice/job_status.js",
    "progress_js": "practice/progress.js",
    "recorder_js": "practice/recorder.js",
}


def static_version(_request):
    cache_key = "practice_static_file_versions"
    version = cache.get(cache_key)
    if version:
        return {"static_version": version}

    version = {}
    for name, static_path in STATIC_VERSION_FILES.items():
        resolved = finders.find(static_path)
        if not resolved:
            version[name] = 0
            continue
        path = Path(resolved)
        version[name] = int(path.stat().st_mtime) if path.exists() else 0

    timeout = (
        settings.STATIC_VERSION_CACHE_TIMEOUT["debug"]
        if settings.DEBUG
        else settings.STATIC_VERSION_CACHE_TIMEOUT["production"]
    )
    cache.set(cache_key, version, timeout)
    return {"static_version": version}
