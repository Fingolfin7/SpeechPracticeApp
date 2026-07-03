from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from django.conf import settings
from django.core.files import File
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand, CommandError
from django.core.exceptions import SuspiciousFileOperation
from django.db import transaction

from practice.models import PracticeSession, ScoringJob


class Command(BaseCommand):
    help = "Copy legacy local audio files to configured storage and update database references."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        dry_run = bool(options["dry_run"])
        if not dry_run and not getattr(settings, "USE_S3", False):
            raise CommandError("Set USE_S3=1 before copying legacy audio to remote storage.")

        refs = set(
            PracticeSession.objects.exclude(audio_path="").values_list("audio_path", flat=True)
        )
        refs.update(
            ScoringJob.objects.exclude(audio_path="").values_list("audio_path", flat=True)
        )

        copied = 0
        already_stored = 0
        missing = 0
        for audio_ref in sorted(refs):
            source = Path(audio_ref)
            if not source.is_file():
                storage_name = str(audio_ref).replace("\\", "/").lstrip("/")
                try:
                    is_stored = not source.is_absolute() and default_storage.exists(storage_name)
                except (OSError, SuspiciousFileOperation, ValueError):
                    is_stored = False
                if is_stored:
                    already_stored += 1
                else:
                    missing += 1
                    self.stderr.write(f"missing: {audio_ref}")
                continue

            target = _target_name(source)
            if dry_run:
                self.stdout.write(f"would copy: {source} -> {target}")
                copied += 1
                continue

            with source.open("rb") as handle:
                stored_name = str(default_storage.save(target, File(handle, name=source.name)))
            with transaction.atomic():
                PracticeSession.objects.filter(audio_path=audio_ref).update(audio_path=stored_name)
                ScoringJob.objects.filter(audio_path=audio_ref).update(audio_path=stored_name)
            copied += 1
            self.stdout.write(f"copied: {source.name} -> {stored_name}")

        self.stdout.write(
            self.style.SUCCESS(
                f"copied={copied} already_stored={already_stored} missing={missing} dry_run={dry_run}"
            )
        )


def _target_name(source: Path) -> str:
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", source.name).strip("-_")
    return PurePosixPath("recordings", "legacy", f"{digest}-{safe_name or 'audio.webm'}").as_posix()
