from __future__ import annotations

from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from .models import (
    GeneratedPracticeScript,
    ImprovementCard,
    PracticeLadder,
    PracticeScript,
    PracticeSession,
    PracticeSettings,
    ScoringJob,
)


class UserIsolationTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.alice = User.objects.create_user(username="alice", password="pass-alice")
        self.bob = User.objects.create_user(username="bob", password="pass-bob")

    def test_history_is_scoped_to_signed_in_user(self):
        PracticeSession.objects.create(
            user=self.alice,
            timestamp="2026-07-02T10:00:00",
            script_name="Alice take",
            script_text="Alice text",
            audio_path="recordings/web/alice/take.webm",
        )
        bob_session = PracticeSession.objects.create(
            user=self.bob,
            timestamp="2026-07-02T11:00:00",
            script_name="Bob take",
            script_text="Bob text",
            audio_path="recordings/web/bob/take.webm",
        )

        self.client.force_login(self.alice)
        response = self.client.get(reverse("practice:sessions"))

        self.assertContains(response, "Alice take")
        self.assertNotContains(response, "Bob take")
        self.assertEqual(
            self.client.get(reverse("practice:session_detail", args=[bob_session.pk])).status_code,
            404,
        )

    def test_scripts_and_cards_are_scoped_but_builtin_scripts_are_shared(self):
        PracticeScript.objects.create(
            title="Shared builtin",
            body="A shared reading.",
            source=PracticeScript.SOURCE_BUILTIN,
        )
        PracticeScript.objects.create(
            user=self.alice,
            title="Alice script",
            body="Alice reading.",
            source=PracticeScript.SOURCE_USER,
        )
        PracticeScript.objects.create(
            user=self.bob,
            title="Bob script",
            body="Bob reading.",
            source=PracticeScript.SOURCE_USER,
        )
        ImprovementCard.objects.create(
            user=self.bob,
            title="Bob focus",
            kind=ImprovementCard.KIND_WORD,
            target_key="bob",
        )

        self.client.force_login(self.alice)
        scripts = self.client.get(reverse("practice:scripts"))
        cards = self.client.get(reverse("practice:cards"))

        self.assertContains(scripts, "Shared builtin")
        self.assertContains(scripts, "Alice script")
        self.assertNotContains(scripts, "Bob script")
        self.assertNotContains(cards, "Bob focus")

    def test_account_settings_and_secrets_are_per_user(self):
        alice_settings = PracticeSettings.load(self.alice)
        bob_settings = PracticeSettings.load(self.bob)

        alice_settings.autumn_project = "Alice project"
        alice_settings.set_secret("openai_api_key", "alice-key")
        alice_settings.set_secret("codex_token_bundle", '{"access_token":"alice-codex"}')
        alice_settings.save()

        bob_settings.refresh_from_db()
        self.assertEqual(bob_settings.autumn_project, "")
        self.assertIsNone(bob_settings.get_secret("openai_api_key"))
        self.assertIsNone(bob_settings.get_secret("codex_token_bundle"))

    def test_signup_creates_private_settings_workspace(self):
        response = self.client.post(
            reverse("practice:signup"),
            {
                "username": "charlie",
                "password1": "strong-pass-charlie-123",
                "password2": "strong-pass-charlie-123",
            },
        )

        self.assertRedirects(response, reverse("practice:dashboard"))
        user = get_user_model().objects.get(username="charlie")
        self.assertTrue(PracticeSettings.objects.filter(user=user).exists())

    def test_claim_existing_data_command_moves_legacy_owner_rows_to_named_user(self):
        User = get_user_model()
        target = User.objects.create_user(username="deployed-user", password="test-pass")
        owner, _created = User.objects.get_or_create(
            username="owner",
            defaults={"is_staff": True, "is_superuser": True},
        )
        owner.set_unusable_password()
        owner.save()
        session = PracticeSession.objects.create(
            user=owner,
            timestamp="2026-07-02T10:00:00",
            script_name="Legacy take",
            script_text="Legacy text",
            audio_path="recordings/web/owner/take.webm",
        )
        card = ImprovementCard.objects.create(
            user=owner,
            title="Legacy focus",
            kind=ImprovementCard.KIND_WORD,
            target_key="legacy",
        )
        script = PracticeScript.objects.create(
            title="Legacy generated",
            body="Legacy drill.",
            source=PracticeScript.SOURCE_GENERATED,
        )
        builtin = PracticeScript.objects.create(
            title="Shared builtin",
            body="Shared.",
            source=PracticeScript.SOURCE_BUILTIN,
        )
        GeneratedPracticeScript.objects.create(user=owner, card=card, script=script)
        ScoringJob.objects.create(
            user=owner,
            script=script,
            card=card,
            script_name=script.title,
            script_text=script.body,
            audio_path="recordings/web/owner/job.webm",
            provider="uploaded_transcript",
        )
        PracticeLadder.objects.create(
            user=owner,
            title="Legacy ladder",
            source=PracticeLadder.SOURCE_GENERATED,
        )
        settings_obj = PracticeSettings.objects.create(user=owner, autumn_project="Legacy")

        dry_run = StringIO()
        call_command("claim_existing_data", "--username", target.username, stdout=dry_run)

        session.refresh_from_db()
        self.assertEqual(session.user, owner)
        self.assertIn("Dry run only", dry_run.getvalue())

        output = StringIO()
        call_command(
            "claim_existing_data",
            "--username",
            target.username,
            "--apply",
            "--delete-empty-placeholder",
            stdout=output,
        )

        session.refresh_from_db()
        card.refresh_from_db()
        script.refresh_from_db()
        settings_obj.refresh_from_db()
        builtin.refresh_from_db()
        self.assertEqual(session.user, target)
        self.assertEqual(card.user, target)
        self.assertEqual(script.user, target)
        self.assertEqual(settings_obj.user, target)
        self.assertIsNone(builtin.user)
        self.assertFalse(User.objects.filter(pk=owner.pk).exists())
