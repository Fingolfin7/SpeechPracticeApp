from django.db import migrations


def pause_phrase_cards(apps, schema_editor):
    ImprovementCard = apps.get_model("practice", "ImprovementCard")
    ImprovementCard.objects.filter(kind="phrase").exclude(status="paused").update(status="paused")


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("practice", "0014_alter_generatedpracticescript_user_and_more"),
    ]

    operations = [
        migrations.RunPython(pause_phrase_cards, noop_reverse),
    ]
