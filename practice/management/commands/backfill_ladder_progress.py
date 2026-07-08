from __future__ import annotations

from collections import defaultdict

from django.core.management.base import BaseCommand

from practice.models import LadderStepProgress, PracticeSession, ScoringJob
from practice.services.jobs import upsert_ladder_step_progress


class Command(BaseCommand):
    help = "Rebuild ladder step progress from historical succeeded scoring jobs."

    def add_arguments(self, parser):
        parser.add_argument("--user", dest="username", help="Only backfill jobs for this username.")

    def handle(self, *args, **options):
        username = options.get("username")
        jobs = (
            ScoringJob.objects.select_related("user", "script")
            .filter(
                status=ScoringJob.STATUS_SUCCEEDED,
                mode=ScoringJob.MODE_SCORE,
                legacy_session_id__isnull=False,
                script__isnull=False,
                script__ladder_steps__isnull=False,
            )
            .distinct()
        )
        if username:
            jobs = jobs.filter(user__username=username)

        entries = []
        affected_by_user: dict[int, set[int]] = defaultdict(set)
        for job in jobs:
            session = PracticeSession.objects.filter(user=job.user, pk=job.legacy_session_id).first()
            if session is None:
                continue
            steps = list(job.script.ladder_steps.select_related("ladder").order_by("ladder_id", "level"))
            if not steps:
                continue
            event_time = job.finished_at or job.created_at
            entries.append((event_time, job.pk, job, session, steps))
            for step in steps:
                affected_by_user[job.user_id].add(step.pk)

        for user_id, step_ids in affected_by_user.items():
            LadderStepProgress.objects.filter(user_id=user_id, step_id__in=step_ids).delete()

        created_count = 0
        pass_count = 0
        for event_time, _job_pk, job, session, steps in sorted(entries, key=lambda item: (item[0], item[1])):
            for step in steps:
                _progress, created, passed = upsert_ladder_step_progress(
                    user=job.user,
                    step=step,
                    session=session,
                    event_time=event_time,
                )
                created_count += int(created)
                pass_count += int(passed)

        self.stdout.write(f"Backfilled ladder progress: rows created {created_count}, passes {pass_count}.")
