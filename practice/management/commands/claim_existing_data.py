from __future__ import annotations

from dataclasses import dataclass

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Count, Q

from practice.models import (
    GeneratedPracticeScript,
    ImprovementCard,
    PracticeLadder,
    PracticeReview,
    PracticeScript,
    PracticeSession,
    PracticeSettings,
    ScoringJob,
    SessionError,
)


@dataclass(frozen=True)
class MovePlan:
    label: str
    count: int


class Command(BaseCommand):
    help = "Assign migrated/legacy SpeechPractice data to a specific existing user."

    def add_arguments(self, parser):
        parser.add_argument("--username", required=True, help="Target Django username.")
        parser.add_argument(
            "--from-username",
            action="append",
            default=[],
            help="Source username to move from. Can be passed more than once.",
        )
        parser.add_argument(
            "--all-users",
            action="store_true",
            help="Move data from every user except the target. Use only for single-user deployments.",
        )
        parser.add_argument(
            "--replace-settings",
            action="store_true",
            help="Replace the target user's settings row with the source settings row if both exist.",
        )
        parser.add_argument(
            "--delete-empty-placeholder",
            action="store_true",
            help="Delete an emptied unusable owner placeholder account after applying.",
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Actually write changes. Without this flag the command only prints a dry run.",
        )

    def handle(self, *args, **options):
        User = get_user_model()
        username = str(options["username"]).strip()
        target = User.objects.filter(username=username).first()
        if target is None:
            raise CommandError(f"User not found: {username}")

        source_users = self._source_users(
            User,
            target=target,
            usernames=[str(item).strip() for item in options["from_username"] if str(item).strip()],
            all_users=bool(options["all_users"]),
        )
        source_usernames = [user.get_username() for user in source_users]

        conflicts = self._card_conflicts(target, source_users)
        if conflicts:
            joined = ", ".join(f"{kind}:{target_key}" for kind, target_key in conflicts[:8])
            raise CommandError(
                "Cannot move cards because the target already has matching card targets: "
                f"{joined}. Merge or delete those cards first."
            )

        plans = self._plans(target, source_users)
        self.stdout.write(f"Target user: {target.get_username()} (id={target.pk})")
        self.stdout.write(f"Source users: {', '.join(source_usernames) if source_usernames else '[none]'}")
        for plan in plans:
            self.stdout.write(f"{plan.label}: {plan.count}")

        source_settings = list(PracticeSettings.objects.filter(user__in=source_users).order_by("id"))
        target_settings = PracticeSettings.objects.filter(user=target).first()
        if source_settings and target_settings and not options["replace_settings"]:
            self.stdout.write(
                self.style.WARNING(
                    "Settings: target already has settings; source settings will be left unchanged. "
                    "Pass --replace-settings to move the first source settings row onto the target."
                )
            )
        elif source_settings:
            self.stdout.write(f"Settings to move: source settings #{source_settings[0].pk}")

        if not options["apply"]:
            self.stdout.write(self.style.WARNING("Dry run only. Re-run with --apply to write changes."))
            return

        with transaction.atomic():
            self._apply(
                target=target,
                source_users=source_users,
                replace_settings=bool(options["replace_settings"]),
            )
            if options["delete_empty_placeholder"]:
                self._delete_empty_placeholders(source_users)

        self.stdout.write(self.style.SUCCESS("Existing SpeechPractice data claimed."))

    def _source_users(self, User, *, target, usernames: list[str], all_users: bool):
        if all_users and usernames:
            raise CommandError("Use either --all-users or --from-username, not both.")
        if all_users:
            return list(User.objects.exclude(pk=target.pk).order_by("id"))
        if usernames:
            users = list(User.objects.filter(username__in=usernames).exclude(pk=target.pk).order_by("id"))
            found = {user.get_username() for user in users}
            missing = sorted(set(usernames) - found)
            if missing:
                raise CommandError(f"Source user(s) not found: {', '.join(missing)}")
            return users

        placeholder = (
            User.objects.filter(username="owner", is_superuser=True)
            .exclude(pk=target.pk)
            .order_by("id")
            .first()
        )
        if placeholder is not None and not placeholder.has_usable_password():
            return [placeholder]
        return []

    def _plans(self, target, source_users: list) -> list[MovePlan]:
        source_filter = Q(user__in=source_users)
        return [
            MovePlan("Sessions", PracticeSession.objects.filter(source_filter).count()),
            MovePlan("Session errors", SessionError.objects.filter(source_filter).count()),
            MovePlan(
                "User scripts",
                PracticeScript.objects.filter(source_filter)
                .exclude(source=PracticeScript.SOURCE_BUILTIN)
                .count(),
            ),
            MovePlan(
                "Unowned legacy scripts",
                PracticeScript.objects.filter(user__isnull=True)
                .exclude(source=PracticeScript.SOURCE_BUILTIN)
                .count(),
            ),
            MovePlan("Cards", ImprovementCard.objects.filter(source_filter).count()),
            MovePlan("Reviews", PracticeReview.objects.filter(source_filter).count()),
            MovePlan("Scoring jobs", ScoringJob.objects.filter(source_filter).count()),
            MovePlan("Generated script records", GeneratedPracticeScript.objects.filter(source_filter).count()),
            MovePlan(
                "Generated/user ladders",
                PracticeLadder.objects.filter(source_filter)
                .exclude(source=PracticeLadder.SOURCE_BUILTIN)
                .count(),
            ),
            MovePlan(
                "Unowned legacy ladders",
                PracticeLadder.objects.filter(user__isnull=True)
                .exclude(source=PracticeLadder.SOURCE_BUILTIN)
                .count(),
            ),
        ]

    def _card_conflicts(self, target, source_users: list) -> list[tuple[str, str]]:
        if not source_users:
            return []
        source_pairs = set(
            ImprovementCard.objects.filter(user__in=source_users).values_list("kind", "target_key")
        )
        target_pairs = set(
            ImprovementCard.objects.filter(user=target).values_list("kind", "target_key")
        )
        return sorted(source_pairs.intersection(target_pairs))

    def _apply(self, *, target, source_users: list, replace_settings: bool):
        for model in (
            PracticeSession,
            SessionError,
            ImprovementCard,
            PracticeReview,
            ScoringJob,
            GeneratedPracticeScript,
        ):
            model.objects.filter(user__in=source_users).update(user=target)

        PracticeScript.objects.filter(user__in=source_users).exclude(
            source=PracticeScript.SOURCE_BUILTIN
        ).update(user=target)
        PracticeScript.objects.filter(user__isnull=True).exclude(
            source=PracticeScript.SOURCE_BUILTIN
        ).update(user=target)

        PracticeLadder.objects.filter(user__in=source_users).exclude(
            source=PracticeLadder.SOURCE_BUILTIN
        ).update(user=target)
        PracticeLadder.objects.filter(user__isnull=True).exclude(
            source=PracticeLadder.SOURCE_BUILTIN
        ).update(user=target)

        self._move_settings(target, source_users, replace_settings=replace_settings)

    def _move_settings(self, target, source_users: list, *, replace_settings: bool):
        source_settings = list(PracticeSettings.objects.filter(user__in=source_users).order_by("id"))
        if not source_settings:
            PracticeSettings.objects.get_or_create(user=target)
            return

        target_settings = PracticeSettings.objects.filter(user=target).first()
        settings_to_move = source_settings[0]
        if target_settings is not None:
            if not replace_settings:
                return
            target_settings.delete()

        settings_to_move.user = target
        settings_to_move.save(update_fields=["user", "updated_at"])
        PracticeSettings.objects.filter(user__in=source_users).exclude(pk=settings_to_move.pk).delete()

    def _delete_empty_placeholders(self, source_users: list) -> None:
        for user in source_users:
            if user.get_username() != "owner" or user.has_usable_password():
                continue
            has_data = (
                PracticeSession.objects.filter(user=user).exists()
                or SessionError.objects.filter(user=user).exists()
                or PracticeScript.objects.filter(user=user).exists()
                or ImprovementCard.objects.filter(user=user).exists()
                or PracticeReview.objects.filter(user=user).exists()
                or ScoringJob.objects.filter(user=user).exists()
                or GeneratedPracticeScript.objects.filter(user=user).exists()
                or PracticeLadder.objects.filter(user=user).exists()
                or PracticeSettings.objects.filter(user=user).exists()
            )
            if not has_data:
                user.delete()
