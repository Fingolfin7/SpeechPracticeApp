from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

from django.core.files.storage import default_storage
from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone


def save_uploaded_audio(audio: UploadedFile, script_title: str) -> str:
    """Stream an upload to the configured storage and return its stable key."""
    extension = safe_audio_extension(audio.name)
    stamp = timezone.localtime().strftime("%Y%m%d_%H%M%S_%f")
    title_slug = re.sub(r"[^a-zA-Z0-9]+", "-", script_title).strip("-").lower()[:40]
    name = PurePosixPath(
        "recordings",
        "web",
        f"{stamp}_{title_slug or 'practice'}{extension}",
    ).as_posix()
    return str(default_storage.save(name, audio))


def safe_audio_extension(filename: str) -> str:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".webm"
    return suffix if suffix in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".txt"} else ".webm"


def audio_exists(audio_ref: str | None) -> bool:
    if not audio_ref:
        return False
    local_path = _existing_local_path(audio_ref)
    if local_path is not None:
        return True
    try:
        return bool(default_storage.exists(_storage_name(audio_ref)))
    except (OSError, ValueError):
        return False


def audio_size(audio_ref: str) -> int:
    local_path = _existing_local_path(audio_ref)
    if local_path is not None:
        return local_path.stat().st_size
    return int(default_storage.size(_storage_name(audio_ref)))


def open_audio(audio_ref: str, mode: str = "rb") -> BinaryIO:
    local_path = _existing_local_path(audio_ref)
    if local_path is not None:
        return local_path.open(mode)
    return default_storage.open(_storage_name(audio_ref), mode)


def delete_audio(audio_ref: str | None) -> bool:
    if not audio_ref:
        return False
    local_path = _existing_local_path(audio_ref)
    if local_path is not None:
        try:
            local_path.unlink()
            return True
        except OSError:
            return False
    name = _storage_name(audio_ref)
    try:
        if not default_storage.exists(name):
            return False
        default_storage.delete(name)
        return True
    except (OSError, ValueError):
        return False


@contextmanager
def materialized_audio(audio_ref: str) -> Iterator[str]:
    """Yield a local path for libraries that cannot read Django storage files."""
    local_path = _existing_local_path(audio_ref)
    if local_path is not None:
        yield str(local_path)
        return

    name = _storage_name(audio_ref)
    try:
        storage_path = Path(default_storage.path(name))
    except (NotImplementedError, AttributeError):
        storage_path = None
    if storage_path is not None and storage_path.is_file():
        yield str(storage_path)
        return

    suffix = Path(name).suffix or ".webm"
    temporary = tempfile.NamedTemporaryFile(
        prefix="speechpractice-audio-",
        suffix=suffix,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary, default_storage.open(name, "rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                temporary.write(chunk)
        yield str(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _existing_local_path(audio_ref: str) -> Path | None:
    path = Path(audio_ref)
    if path.is_file():
        return path
    return None


def _storage_name(audio_ref: str) -> str:
    return str(audio_ref).replace("\\", "/").lstrip("/")
