from django.db import migrations, models


def replace_removed_model_selections(apps, schema_editor):
    PracticeSettings = apps.get_model("practice", "PracticeSettings")
    PracticeSettings.objects.filter(
        openai_script_model__in=["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
    ).update(openai_script_model="gpt-5.6-luna")
    PracticeSettings.objects.filter(
        anthropic_script_model="claude-sonnet-4-6"
    ).update(anthropic_script_model="claude-sonnet-5")


class Migration(migrations.Migration):

    dependencies = [
        ("practice", "0017_practicereview_story_data"),
    ]

    operations = [
        migrations.AlterField(
            model_name="practicesettings",
            name="openai_script_model",
            field=models.CharField(default="gpt-5.6-luna", max_length=96),
        ),
        migrations.AlterField(
            model_name="practicesettings",
            name="anthropic_script_model",
            field=models.CharField(default="claude-sonnet-5", max_length=96),
        ),
        migrations.RunPython(
            replace_removed_model_selections,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
