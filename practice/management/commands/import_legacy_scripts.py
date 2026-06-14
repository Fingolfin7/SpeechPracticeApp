from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from practice.models import PracticeScript


class Command(BaseCommand):
    help = "Import .txt scripts from the legacy scripts directory into Django."

    def add_arguments(self, parser):
        parser.add_argument("--source", default=str(settings.LEGACY_SCRIPTS_DIR))
        parser.add_argument("--replace", action="store_true")

    def handle(self, *args, **options):
        source = Path(options["source"])
        if not source.exists():
            self.stderr.write(self.style.ERROR(f"Scripts directory not found: {source}"))
            return

        imported = 0
        updated = 0
        for path in sorted(source.glob("*.txt")):
            title = path.stem
            body = path.read_text(encoding="utf-8").strip()
            if not body:
                continue
            defaults = {
                "body": body,
                "source": PracticeScript.SOURCE_BUILTIN,
                "source_ref": str(path),
                "tags": ["legacy", "poem"],
                "active": True,
            }
            script, created = PracticeScript.objects.update_or_create(
                title=title,
                defaults=defaults if options["replace"] else {**defaults, "body": body},
            )
            imported += 1 if created else 0
            updated += 0 if created else 1

        self.stdout.write(
            self.style.SUCCESS(f"Imported {imported} scripts; updated {updated}.")
        )
