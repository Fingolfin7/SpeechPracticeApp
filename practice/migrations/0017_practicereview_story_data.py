import re

from django.db import migrations, models

# Inlined copy of practice.services.review_notes.parse_legacy_review_notes so this
# migration never depends on app code that may move or change.
_LEGACY_REVIEW_NOTES_RE = re.compile(
    r"Quality (\d+(?:\.\d+)?); mastery (\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?)"
)


def parse_legacy_review_notes(notes):
    match = _LEGACY_REVIEW_NOTES_RE.search(notes or "")
    if not match:
        return None
    return float(match.group(1)), float(match.group(3))


def backfill_review_story_data(apps, schema_editor):
    PracticeReview = apps.get_model("practice", "PracticeReview")
    pending = []
    for review in PracticeReview.objects.exclude(notes="").iterator(chunk_size=500):
        parsed = parse_legacy_review_notes(review.notes)
        if parsed is None:
            continue
        review.quality, review.mastery_after = parsed
        pending.append(review)
        if len(pending) >= 500:
            PracticeReview.objects.bulk_update(pending, ["quality", "mastery_after"])
            pending = []
    if pending:
        PracticeReview.objects.bulk_update(pending, ["quality", "mastery_after"])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("practice", "0016_ladder_step_progress"),
    ]

    operations = [
        migrations.AddField(
            model_name="practicereview",
            name="quality",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="practicereview",
            name="mastery_after",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="practicereview",
            name="evidence",
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.RunPython(backfill_review_story_data, noop_reverse),
    ]
