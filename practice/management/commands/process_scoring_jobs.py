from __future__ import annotations

from django.core.management.base import BaseCommand

from practice.models import ScoringJob
from practice.services.jobs import process_next_scoring_job


class Command(BaseCommand):
    help = "Process queued SpeechPractice scoring jobs."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=1)

    def handle(self, *args, **options):
        limit = max(1, int(options["limit"]))
        processed = 0
        for _ in range(limit):
            job = process_next_scoring_job()
            if job is None:
                break
            processed += 1
            self.stdout.write(f"Processed job #{job.pk}: {job.get_status_display()}")
        queued = ScoringJob.objects.filter(status=ScoringJob.STATUS_QUEUED).count()
        self.stdout.write(self.style.SUCCESS(f"processed={processed} queued={queued}"))
