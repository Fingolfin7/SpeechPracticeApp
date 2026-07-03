from __future__ import annotations

import time

from django.core.management.base import BaseCommand

from practice.models import ScoringJob
from practice.services.jobs import process_next_scoring_job, recover_stale_scoring_jobs


class Command(BaseCommand):
    help = "Process queued SpeechPractice scoring jobs."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=1)
        parser.add_argument("--watch", action="store_true")
        parser.add_argument("--poll-seconds", type=float, default=2.0)
        parser.add_argument("--stale-after-minutes", type=int, default=30)

    def handle(self, *args, **options):
        recovered = recover_stale_scoring_jobs(
            stale_after_minutes=int(options["stale_after_minutes"]),
        )
        if recovered:
            self.stdout.write(self.style.WARNING(f"recovered={recovered}"))

        if options["watch"]:
            self._watch(max(0.5, float(options["poll_seconds"])))
            return

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

    def _watch(self, poll_seconds: float) -> None:
        self.stdout.write(self.style.SUCCESS(f"worker=ready poll_seconds={poll_seconds:g}"))
        try:
            while True:
                job = process_next_scoring_job()
                if job is None:
                    time.sleep(poll_seconds)
                else:
                    self.stdout.write(
                        f"Processed job #{job.pk}: {job.get_status_display()}"
                    )
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("worker=stopped"))
