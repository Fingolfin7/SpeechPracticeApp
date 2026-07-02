from __future__ import annotations

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from practice.services.analytics import refresh_improvement_cards


class Command(BaseCommand):
    help = "Refresh improvement cards from recent scoring history."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--username", default="")

    def handle(self, *args, **options):
        users = get_user_model().objects.all().order_by("id")
        username = (options["username"] or "").strip()
        if username:
            users = users.filter(username=username)
        total = 0
        count = 0
        for user in users:
            created = refresh_improvement_cards(user=user, days=options["days"])
            total += created
            count += 1
            self.stdout.write(f"{user.get_username()}: {created} new cards created.")
        self.stdout.write(self.style.SUCCESS(f"Cards refreshed for {count} users. {total} new cards created."))
