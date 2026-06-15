from __future__ import annotations

from django import forms
from django.conf import settings

from .models import ImprovementCard, PracticeScript, PracticeSession, PracticeSettings
from .services.transcription import provider_label


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={"multiple": True}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        if not data:
            return []
        files = data if isinstance(data, (list, tuple)) else [data]
        return [super(MultipleFileField, self).clean(file, initial) for file in files]


class PracticeScriptForm(forms.ModelForm):
    tags_text = forms.CharField(
        required=False,
        label="Tags",
        help_text="Comma-separated tags, such as poem, breath, final consonants.",
    )

    class Meta:
        model = PracticeScript
        fields = ["title", "author", "body", "source", "difficulty", "active"]
        widgets = {
            "body": forms.Textarea(attrs={"rows": 14}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            self.fields["tags_text"].initial = ", ".join(self.instance.tags or [])

    def save(self, commit=True):
        instance = super().save(commit=False)
        tags_raw = self.cleaned_data.get("tags_text") or ""
        instance.tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        if commit:
            instance.save()
            self.save_m2m()
        return instance


class PracticeRunForm(forms.Form):
    MODE_SCRIPT = "script"
    MODE_QUICK = "quick"
    MODE_FREE = "free_speak"
    MODE_CHOICES = [
        (MODE_SCRIPT, "Script"),
        (MODE_QUICK, "Quick Practice"),
        (MODE_FREE, "Free Speak"),
    ]

    mode = forms.ChoiceField(
        required=False,
        choices=MODE_CHOICES,
        initial=MODE_SCRIPT,
        widget=forms.RadioSelect,
        label="Recording mode",
    )
    script = forms.ModelChoiceField(
        queryset=PracticeScript.objects.none(),
        label="Practice script",
        required=False,
    )
    card = forms.ModelChoiceField(
        queryset=ImprovementCard.objects.none(),
        required=False,
        widget=forms.HiddenInput,
    )
    audio = forms.FileField(
        required=False,
        label="Recording",
        help_text="Record in the browser or upload an audio file.",
    )
    provider = forms.ChoiceField(
        required=False,
        choices=[],
        label="Transcription provider",
    )

    def __init__(self, *args, **kwargs):
        initial_script = kwargs.pop("initial_script", None)
        initial_card = kwargs.pop("initial_card", None)
        super().__init__(*args, **kwargs)
        self.fields["script"].queryset = PracticeScript.objects.filter(active=True)
        self.fields["card"].queryset = ImprovementCard.objects.exclude(
            status=ImprovementCard.STATUS_PAUSED
        )
        if initial_script is not None:
            self.fields["script"].initial = initial_script
        elif not self.is_bound:
            self.fields["script"].initial = self.fields["script"].queryset.first()
        if initial_card is not None:
            self.fields["card"].initial = initial_card
        provider_choices = [
            ("local_whisper", provider_label("local_whisper")),
            ("openai", provider_label("openai")),
            ("uploaded_transcript", provider_label("uploaded_transcript")),
        ]
        self.fields["provider"].choices = provider_choices
        self.fields["provider"].initial = "local_whisper"

    def clean(self):
        cleaned = super().clean()
        mode = cleaned.get("mode") or self.MODE_SCRIPT
        script = cleaned.get("script")
        if mode != self.MODE_FREE and script is None:
            self.add_error("script", "Choose a script for scripted practice.")
        return cleaned


class BulkScriptImportForm(forms.Form):
    files = MultipleFileField(
        label="Script files",
        help_text="Upload .txt, .md, .csv, .json, or .zip files.",
    )
    default_author = forms.CharField(
        required=False,
        label="Default author",
        help_text="Used when files do not include an author/poet field.",
    )
    tags_text = forms.CharField(
        required=False,
        label="Tags",
        help_text="Comma-separated tags added to every imported script.",
        initial="imported",
    )
    replace = forms.BooleanField(
        required=False,
        label="Replace matching scripts",
        initial=True,
    )

    def tags(self) -> list[str]:
        raw = self.cleaned_data.get("tags_text") or ""
        return [tag.strip() for tag in raw.split(",") if tag.strip()]


class TranscriptEditForm(forms.ModelForm):
    class Meta:
        model = PracticeSession
        fields = ["transcript"]
        widgets = {
            "transcript": forms.Textarea(attrs={"rows": 8}),
        }


class AccountSettingsForm(forms.ModelForm):
    openai_api_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(
            render_value=False,
            attrs={"autocomplete": "new-password", "placeholder": "OpenAI API key"},
        ),
    )
    anthropic_api_key = forms.CharField(
        required=False,
        label="Anthropic API key",
        widget=forms.PasswordInput(
            render_value=False,
            attrs={"autocomplete": "new-password", "placeholder": "Anthropic API key"},
        ),
    )
    autumn_token = forms.CharField(
        required=False,
        label="Autumn token",
        widget=forms.PasswordInput(
            render_value=False,
            attrs={"autocomplete": "new-password", "placeholder": "Autumn API token"},
        ),
    )
    clear_openai_api_key = forms.BooleanField(required=False, label="Clear OpenAI key")
    clear_anthropic_api_key = forms.BooleanField(required=False, label="Clear Anthropic key")
    clear_autumn_token = forms.BooleanField(required=False, label="Clear Autumn token")
    autumn_subprojects_text = forms.CharField(
        required=False,
        label="Autumn subprojects",
        help_text="Comma-separated Autumn subprojects.",
    )

    class Meta:
        model = PracticeSettings
        fields = [
            "transcription_provider",
            "script_generation_provider",
            "openai_script_model",
            "anthropic_script_model",
            "openai_transcription_model",
            "whisper_model_name",
            "whisper_device",
            "whisper_preset",
            "whisper_language",
            "whisper_timestamps",
            "whisper_beam_size",
            "whisper_temperature",
            "whisper_no_speech_threshold",
            "whisper_condition_on_previous_text",
            "whisper_chunk_seconds",
            "autumn_base_url",
            "autumn_project",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["openai_script_model"].widget = forms.Select(
            choices=settings.OPENAI_SCRIPT_MODEL_CHOICES
        )
        self.fields["anthropic_script_model"].widget = forms.Select(
            choices=settings.ANTHROPIC_SCRIPT_MODEL_CHOICES
        )
        self.fields["whisper_model_name"].widget = forms.Select(
            choices=[
                ("tiny.en", "tiny.en - fastest English"),
                ("base.en", "base.en - fast English"),
                ("small.en", "small.en - balanced English"),
                ("medium.en", "medium.en - higher accuracy English"),
                ("tiny", "tiny - multilingual"),
                ("base", "base - multilingual"),
                ("small", "small - multilingual"),
                ("medium", "medium - multilingual"),
            ]
        )
        self.fields["whisper_beam_size"].widget.attrs.update({"min": 1, "max": 10})
        self.fields["whisper_temperature"].widget.attrs.update({"min": 0, "max": 1, "step": 0.05})
        self.fields["whisper_no_speech_threshold"].widget.attrs.update({"min": 0, "max": 1, "step": 0.05})
        self.fields["whisper_chunk_seconds"].widget.attrs.update({"min": 10, "max": 600, "step": 10})
        self.fields["openai_api_key"].initial = ""
        self.fields["anthropic_api_key"].initial = ""
        self.fields["autumn_token"].initial = ""
        if self.instance and self.instance.pk:
            self.fields["autumn_subprojects_text"].initial = ", ".join(
                self.instance.autumn_subprojects or []
            )

    def save(self, commit=True):
        instance = super().save(commit=False)
        if not instance.autumn_base_url:
            instance.autumn_base_url = settings.DEFAULT_AUTUMN_BASE_URL
        subs_raw = self.cleaned_data.get("autumn_subprojects_text") or ""
        instance.autumn_subprojects = [
            item.strip() for item in subs_raw.split(",") if item.strip()
        ]
        for field in ("openai_api_key", "anthropic_api_key", "autumn_token"):
            if self.cleaned_data.get(f"clear_{field}"):
                instance.set_secret(field, None)
            elif self.cleaned_data.get(field):
                instance.set_secret(field, self.cleaned_data[field].strip())
        if commit:
            instance.save()
        return instance
