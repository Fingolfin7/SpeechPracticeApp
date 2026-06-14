from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from practice.models import ImprovementCard, PracticeSettings


@dataclass(frozen=True)
class GeneratedScriptDraft:
    title: str
    body: str
    prompt_snapshot: str
    provider: str


class ScriptGenerationProvider(Protocol):
    name: str

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        ...


def build_generation_prompt(card: ImprovementCard) -> str:
    return (
        "Create a short speech-practice script for one learner.\n"
        f"Focus area: {card.title}\n"
        f"Target key: {card.target_key}\n"
        f"Reason: {card.prompt}\n"
        f"Recent evidence: {card.stats}\n"
        "Use 8 short lines. Start easy, then increase difficulty. "
        "Keep it natural enough to say out loud.\n"
        "Return exactly this format:\n"
        "TITLE: <short title>\n"
        "SCRIPT:\n"
        "<line 1>\n"
        "<line 2>\n"
        "..."
    )


class LocalTemplateScriptProvider:
    name = "local_template"

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        target = card.target_key
        title = f"Drill: {card.title}"
        if card.kind == ImprovementCard.KIND_PHRASE:
            body = "\n".join(
                [
                    f"{target}.",
                    f"I say {target} with an even pace.",
                    f"I pause, breathe, and repeat: {target}.",
                    f"The words around {target} stay relaxed.",
                    f"Before the phrase, I stay steady; {target}.",
                    f"After the phrase, I keep the same clear rhythm.",
                    f"I can place {target} inside a longer sentence without rushing.",
                    f"One clean phrase becomes the next clean phrase.",
                ]
            )
            prompt = build_generation_prompt(card)
            return GeneratedScriptDraft(
                title=title,
                body=body,
                prompt_snapshot=prompt,
                provider=self.name,
            )
        body = "\n".join(
            [
                f"I notice {target} clearly and calmly.",
                f"Today I practice {target} with steady breath.",
                f"The phrase stays crisp when {target} appears early.",
                f"The phrase stays crisp when {target} appears late.",
                f"I slow down, reset, and say {target} again.",
                f"Short words, long words, and {target} all stay clear.",
                f"I keep my pace even while the sentence grows around {target}.",
                f"One clean repetition becomes the next clean repetition.",
            ]
        )
        prompt = build_generation_prompt(card)
        return GeneratedScriptDraft(
            title=title,
            body=body,
            prompt_snapshot=prompt,
            provider=self.name,
        )


class OpenAIScriptProvider:
    name = "openai"

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        app_settings = _practice_settings()
        api_key = (app_settings.get_secret("openai_api_key") if app_settings else None) or settings.OPENAI_API_KEY
        model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI script generation.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI script generation.") from exc

        prompt = build_generation_prompt(card)
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise speech coach. Generate concise, speakable "
                        "practice material. Do not include analysis or markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=700,
        )
        text = getattr(response, "output_text", "") or str(response)
        title, body = parse_generated_script(text, fallback_title=f"Drill: {card.title}")
        return GeneratedScriptDraft(
            title=title,
            body=body,
            prompt_snapshot=prompt,
            provider=f"openai:{model}",
        )


class AnthropicScriptProvider:
    name = "anthropic"

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        app_settings = _practice_settings()
        api_key = (app_settings.get_secret("anthropic_api_key") if app_settings else None) or settings.ANTHROPIC_API_KEY
        model = (app_settings.anthropic_script_model if app_settings else None) or settings.ANTHROPIC_SCRIPT_MODEL
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic script generation.")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the optional 'anthropic' package to use Anthropic script generation.") from exc

        prompt = build_generation_prompt(card)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=700,
            system=(
                "You are a precise speech coach. Generate concise, speakable "
                "practice material. Do not include analysis or markdown."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        title, body = parse_generated_script("\n".join(parts), fallback_title=f"Drill: {card.title}")
        return GeneratedScriptDraft(
            title=title,
            body=body,
            prompt_snapshot=prompt,
            provider=f"anthropic:{model}",
        )


def generate_local_template(card: ImprovementCard) -> GeneratedScriptDraft:
    return LocalTemplateScriptProvider().generate(card)


def generate_script_draft(
    card: ImprovementCard,
    provider_name: str | None = None,
) -> GeneratedScriptDraft:
    return get_script_generation_provider(provider_name).generate(card)


def get_script_generation_provider(provider_name: str | None = None) -> ScriptGenerationProvider:
    app_settings = _practice_settings()
    provider = provider_name or (app_settings.script_generation_provider if app_settings else None) or settings.SCRIPT_GENERATION_PROVIDER
    if provider == "local_template":
        return LocalTemplateScriptProvider()
    if provider == "openai":
        return OpenAIScriptProvider()
    if provider == "anthropic":
        return AnthropicScriptProvider()
    raise ValueError(f"Unknown script generation provider: {provider}")


def script_generation_provider_choices() -> list[tuple[str, str]]:
    app_settings = _practice_settings()
    openai_model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
    anthropic_model = (app_settings.anthropic_script_model if app_settings else None) or settings.ANTHROPIC_SCRIPT_MODEL
    return [
        ("local_template", "Local template"),
        ("openai", f"OpenAI ({openai_model})"),
        ("anthropic", f"Anthropic ({anthropic_model})"),
    ]


def _practice_settings() -> PracticeSettings | None:
    try:
        return PracticeSettings.load()
    except (OperationalError, ProgrammingError):
        return None


def parse_generated_script(text: str, fallback_title: str) -> tuple[str, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return fallback_title, "Read this line slowly and clearly."

    title = fallback_title
    body = cleaned
    lines = [line.rstrip() for line in cleaned.splitlines()]
    for idx, line in enumerate(lines):
        upper = line.upper()
        if upper.startswith("TITLE:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                title = candidate[:255]
        if upper.startswith("SCRIPT:"):
            body = "\n".join(lines[idx + 1 :]).strip()
            break
    if not body:
        body = "\n".join(line for line in lines if not line.upper().startswith("TITLE:")).strip()
    return title, body or cleaned
