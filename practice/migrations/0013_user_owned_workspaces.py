import os

from django.conf import settings
from django.contrib.auth.hashers import make_password
from django.db import migrations, models
import django.db.models.deletion


def _owner_user(apps):
    app_label, model_name = settings.AUTH_USER_MODEL.split(".")
    User = apps.get_model(app_label, model_name)
    owner_username = os.getenv("SPEECHPRACTICE_OWNER_USERNAME", "").strip()
    user = User.objects.filter(username=owner_username).first() if owner_username else None
    if owner_username and user is None:
        raise RuntimeError(f"SPEECHPRACTICE_OWNER_USERNAME does not exist: {owner_username}")
    if user is None:
        user = User.objects.filter(is_superuser=True).order_by("id").first()
    if user is None:
        user = User.objects.order_by("id").first()
    if user is None:
        user = User(
            username="owner",
            is_staff=True,
            is_superuser=True,
            password=make_password(None),
        )
        user.save()
    return user


def assign_existing_rows(apps, schema_editor):
    owner = _owner_user(apps)
    PracticeSession = apps.get_model("practice", "PracticeSession")
    SessionError = apps.get_model("practice", "SessionError")
    PracticeScript = apps.get_model("practice", "PracticeScript")
    ImprovementCard = apps.get_model("practice", "ImprovementCard")
    PracticeReview = apps.get_model("practice", "PracticeReview")
    ScoringJob = apps.get_model("practice", "ScoringJob")
    GeneratedPracticeScript = apps.get_model("practice", "GeneratedPracticeScript")
    PracticeLadder = apps.get_model("practice", "PracticeLadder")
    PracticeSettings = apps.get_model("practice", "PracticeSettings")

    PracticeSession.objects.filter(user__isnull=True).update(user=owner)
    ImprovementCard.objects.filter(user__isnull=True).update(user=owner)
    PracticeReview.objects.filter(user__isnull=True).update(user=owner)
    ScoringJob.objects.filter(user__isnull=True).update(user=owner)
    GeneratedPracticeScript.objects.filter(user__isnull=True).update(user=owner)
    PracticeSettings.objects.filter(user__isnull=True).update(user=owner)

    PracticeScript.objects.filter(user__isnull=True).exclude(source="builtin").update(user=owner)
    PracticeLadder.objects.filter(user__isnull=True).exclude(source="builtin").update(user=owner)

    sessions_by_id = dict(PracticeSession.objects.values_list("id", "user_id"))
    for error in SessionError.objects.filter(user__isnull=True).only("id", "session_id"):
        error.user_id = sessions_by_id.get(error.session_id) or owner.pk
        error.save(update_fields=["user"])


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("practice", "0012_practicesession_self_review_notes"),
    ]

    operations = [
        migrations.AddField(
            model_name="practicesession",
            name="user",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_sessions",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="sessionerror",
            name="user",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_session_errors",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="practicescript",
            name="user",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_scripts",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="improvementcard",
            name="user",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="improvement_cards",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="practicereview",
            name="user",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_reviews",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="scoringjob",
            name="user",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="scoring_jobs",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="generatedpracticescript",
            name="user",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="generated_practice_scripts",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="practiceladder",
            name="user",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_ladders",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="practicesettings",
            name="user",
            field=models.OneToOneField(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_settings",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.RunPython(assign_existing_rows, migrations.RunPython.noop),
        migrations.RemoveConstraint(
            model_name="improvementcard",
            name="unique_card_target",
        ),
        migrations.AlterField(
            model_name="practicesession",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_sessions",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="sessionerror",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_session_errors",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="improvementcard",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="improvement_cards",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="practicereview",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_reviews",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="scoringjob",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="scoring_jobs",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="generatedpracticescript",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="generated_practice_scripts",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="practicesettings",
            name="user",
            field=models.OneToOneField(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="practice_settings",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddConstraint(
            model_name="improvementcard",
            constraint=models.UniqueConstraint(
                fields=("user", "kind", "target_key"),
                name="unique_user_card_target",
            ),
        ),
        migrations.AddIndex(
            model_name="practicescript",
            index=models.Index(fields=["user", "source", "active"], name="practice_pr_user_id_239018_idx"),
        ),
        migrations.AddIndex(
            model_name="practicescript",
            index=models.Index(fields=["user", "practice_kind", "active"], name="practice_pr_user_id_c01eb1_idx"),
        ),
        migrations.AddIndex(
            model_name="improvementcard",
            index=models.Index(fields=["user", "kind", "status"], name="practice_im_user_id_d7f238_idx"),
        ),
        migrations.AddIndex(
            model_name="improvementcard",
            index=models.Index(fields=["user", "due_at"], name="practice_im_user_id_24b8ff_idx"),
        ),
        migrations.AddIndex(
            model_name="scoringjob",
            index=models.Index(fields=["user", "status", "created_at"], name="practice_sc_user_id_0c5301_idx"),
        ),
        migrations.AddIndex(
            model_name="practiceladder",
            index=models.Index(fields=["user", "source", "active"], name="practice_pr_user_id_32c3e4_idx"),
        ),
    ]
