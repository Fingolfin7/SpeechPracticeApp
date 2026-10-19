from django.db import migrations

# Sound, character, and position cards are derived from spelling diffs rather
# than audio, and fluency inputs (filled pauses, segment-gap pauses) are mostly
# invisible to Whisper. None can be graded honestly, so they leave the queue.
# Pausing (not deleting) keeps their history and reviews intact.
UNMEASURABLE_KINDS = ("sound", "character", "position", "fluency")


def pause_unmeasurable_cards(apps, schema_editor):
    ImprovementCard = apps.get_model("practice", "ImprovementCard")
    ImprovementCard.objects.filter(kind__in=UNMEASURABLE_KINDS).exclude(status="paused").update(status="paused")


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("practice", "0019_remove_practicesettings_autumn_active_session_id_and_more"),
    ]

    operations = [
        migrations.RunPython(pause_unmeasurable_cards, noop_reverse),
    ]
