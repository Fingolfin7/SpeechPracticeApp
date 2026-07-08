from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


def backfill_min_clarity(apps, schema_editor):
    PracticeLadderStep = apps.get_model("practice", "PracticeLadderStep")
    # Keep this ramp in sync with practice.views.LADDER_MIN_CLARITY_BY_LEVEL.
    ramp = {
        1: 0.85,
        2: 0.88,
        3: 0.90,
        4: 0.92,
        5: 0.95,
    }
    for level, min_clarity in ramp.items():
        PracticeLadderStep.objects.filter(level=level).update(min_clarity=min_clarity)
    PracticeLadderStep.objects.filter(level__gt=5).update(min_clarity=0.95)


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("practice", "0015_pause_phrase_improvement_cards"),
    ]

    operations = [
        migrations.AddField(
            model_name="practiceladderstep",
            name="min_clarity",
            field=models.FloatField(default=0.9),
        ),
        migrations.CreateModel(
            name="LadderStepProgress",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("best_clarity", models.FloatField(default=0.0)),
                ("attempts", models.PositiveIntegerField(default=0)),
                ("passed_at", models.DateTimeField(blank=True, null=True)),
                ("best_session_id", models.IntegerField(blank=True, null=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "step",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="progress",
                        to="practice.practiceladderstep",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="ladder_step_progress",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="ladderstepprogress",
            constraint=models.UniqueConstraint(
                fields=("user", "step"),
                name="unique_user_ladder_step_progress",
            ),
        ),
        migrations.AddIndex(
            model_name="ladderstepprogress",
            index=models.Index(fields=["user", "step"], name="practice_la_user_id_fdbc10_idx"),
        ),
        migrations.RunPython(backfill_min_clarity, noop_reverse),
    ]
