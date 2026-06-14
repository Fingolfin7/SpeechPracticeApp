from __future__ import annotations

from django.core.management.base import BaseCommand

from practice.services.analytics import refresh_improvement_cards


class Command(BaseCommand):
    help = "Refresh improvement cards from recent scoring history."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)

    def handle(self, *args, **options):
        created = refresh_improvement_cards(days=options["days"])
        self.stdout.write(self.style.SUCCESS(f"Cards refreshed. {created} new cards created."))
