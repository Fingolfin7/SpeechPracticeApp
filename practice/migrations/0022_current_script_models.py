from django.db import migrations, models

# Saved model choices that are no longer offered, mapped to their current
# equivalents. Without this the Account selector would silently show (and on
# the next save, store) the first option instead of what the user had picked.
OPENAI_RENAMES = {
    "gpt-5.6-sol": "gpt-6.1-sol",
    "gpt-5.6-luna": "gpt-6-luna",
    "gpt-5.6-terra": "gpt-6-luna",
    "gpt-5.5": "gpt-6-luna",
}
ANTHROPIC_RENAMES = {
    "claude-sonnet-5": "claude-sonnet-5-5",
    "claude-opus-4-8": "claude-opus-5-5",
    "claude-fable-5": "claude-fable-5-1",
    "claude-haiku-4-5": "claude-haiku-5-5",
}


def remap_models(apps, schema_editor):
    PracticeSettings = apps.get_model("practice", "PracticeSettings")
    for old, new in OPENAI_RENAMES.items():
        PracticeSettings.objects.filter(openai_script_model=old).update(openai_script_model=new)
    for old, new in ANTHROPIC_RENAMES.items():
        PracticeSettings.objects.filter(anthropic_script_model=old).update(anthropic_script_model=new)


class Migration(migrations.Migration):

    dependencies = [
        ("practice", "0021_tts_precheck"),
    ]

    operations = [
        migrations.AlterField(
            model_name="practicesettings",
            name="openai_script_model",
            field=models.CharField(default="gpt-6-luna", max_length=96),
        ),
        migrations.AlterField(
            model_name="practicesettings",
            name="anthropic_script_model",
            field=models.CharField(default="claude-sonnet-5-5", max_length=96),
        ),
        migrations.RunPython(remap_models, migrations.RunPython.noop),
    ]
