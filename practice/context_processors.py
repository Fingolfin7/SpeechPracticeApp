from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.contrib.staticfiles import finders
from django.core.cache import cache


STATIC_VERSION_FILES = {
    "app": "practice/app.css",
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
