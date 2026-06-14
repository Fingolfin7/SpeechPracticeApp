from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from practice.services.script_import import import_script_items, parse_path


class Command(BaseCommand):
    help = "Import scripts from a file, folder, CSV, JSON, or ZIP corpus."

    def add_arguments(self, parser):
        parser.add_argument("source", help="File or directory to import.")
        parser.add_argument("--author", default="", help="Default author when missing.")
        parser.add_argument(
            "--tag",
            action="append",
            default=[],
            help="Extra tag to apply to imported scripts. Can be repeated.",
        )
        parser.add_argument(
            "--no-replace",
            action="store_true",
            help="Skip matching scripts instead of updating them.",
        )

    def handle(self, *args, **options):
        source = Path(options["source"])
        if not source.exists():
            self.stderr.write(self.style.ERROR(f"Import source not found: {source}"))
            return

        items = parse_path(source, default_author=options["author"])
        result = import_script_items(
            items,
            extra_tags=options["tag"],
            replace=not options["no_replace"],
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"created={result.created} updated={result.updated} skipped={result.skipped}"
            )
        )
