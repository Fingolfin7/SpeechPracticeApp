from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from practice.models import ImprovementCard, PracticeSettings
from practice.services.codex_auth import (
    CodexAuthError,
    access_token_expires_soon,
    deserialize_token_bundle,
    refresh_token_bundle,
    serialize_token_bundle,
)


@dataclass(frozen=True)
class GeneratedScriptDraft:
    title: str
    body: str
    prompt_snapshot: str
    provider: str
    auth_source: str = ""


@dataclass(frozen=True)
class LadderLevelDraft:
    level: int
    title: str
    body: str
    focus: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneratedLadderDraft:
    title: str
    theme: str
    levels: tuple[LadderLevelDraft, ...]
    prompt_snapshot: str
    provider: str
    auth_source: str = ""


class ScriptGenerationProvider(Protocol):
    name: str

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        ...

    def generate_ladder(
        self,
        cards: list[ImprovementCard],
        theme: str = "",
    ) -> GeneratedLadderDraft:
        ...


def build_generation_prompt(card: ImprovementCard) -> str:
    return (
        "Create a short speech-practice drill for one learner.\n"
        f"Focus area: {card.title}\n"
        f"Target key: {card.target_key}\n"
        f"Reason: {card.prompt}\n"
        f"Recent evidence: {card.stats}\n"
        "Use the pattern of a leveled tongue-twister ladder: begin with simple, "
        "clean repetitions, then add denser phrases, then one or two harder control "
        "lines. Keep every line speakable out loud. Favor clarity over speed. "
        "Include the target in varied word positions when that makes sense.\n"
        "Use 8 lines total. Do not include coaching notes, bullets, numbering, or markdown.\n"
        "Return exactly this format:\n"
        "TITLE: <short title>\n"
        "SCRIPT:\n"
        "<line 1>\n"
        "<line 2>\n"
        "..."
    )


def build_ladder_generation_prompt(cards: list[ImprovementCard], theme: str = "") -> str:
    card_lines = []
    for card in cards:
        card_lines.append(
            "\n".join(
                [
                    f"- Title: {card.title}",
                    f"  Type: {card.get_kind_display()}",
                    f"  Target: {card.target_key}",
                    f"  Why it matters: {card.prompt}",
                    f"  Recent evidence: {card.stats}",
                    f"  Mastery: {card.mastery:.2f}",
                ]
            )
        )
    theme_line = theme.strip() or "No optional creative theme was provided."
    return (
        "You are generating a five-step speech practice ladder for SpeechPractice.\n"
        "SpeechPractice records the learner reading a target text aloud, transcribes the take, "
        "then scores pronunciation clarity, word errors, character errors, pacing, pauses, and confidence. "
        "The learner wants targeted repetition that irons out recurring speech issues, but the material "
        "is allowed to have personality. Do not optimize for bland coaching copy. If a creative theme is "
        "provided, use it to make the drills more memorable while keeping the target sounds and words central.\n\n"
        f"Creative theme requested by the user: {theme_line}\n\n"
        "Current struggle cards:\n"
        f"{chr(10).join(card_lines) if card_lines else '- No cards were provided; create a general clarity ladder.'}\n\n"
        "Create exactly five levels. Difficulty should rise steadily:\n"
        "1. short warm-up phrases with obvious targets\n"
        "2. short contrast/repetition lines\n"
        "3. denser sentences with target sounds in varied positions\n"
        "4. mixed-focus paragraph-like lines with rhythm and breath control\n"
        "5. performance challenge that is still speakable and scorable\n\n"
        "Return strict JSON only. No markdown, comments, or prose outside JSON. Schema:\n"
        "{\n"
        '  "title": "short ladder title",\n'
        '  "theme": "one sentence describing the target + creative theme",\n'
        '  "levels": [\n'
        '    {"level": 1, "title": "Warm-up", "focus": ["target"], "lines": ["line 1", "line 2", "..."]}\n'
        "  ]\n"
        "}\n"
        "Each level must contain 4 to 8 lines. Keep each line suitable for reading aloud."
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
                    f"{target}, clear and steady.",
                    f"I say {target} with an even pace.",
                    f"Slow breath first, then {target}.",
                    f"The words before {target} stay relaxed.",
                    f"The words after {target} keep the same rhythm.",
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
                auth_source="local",
            )
        body = "\n".join(
            [
                f"{target}.",
                f"{target}, clean and calm.",
                f"I practice {target} with steady breath.",
                f"The phrase stays crisp when {target} appears early.",
                f"The phrase stays crisp when {target} appears late.",
                f"I slow down, reset, and say {target} again.",
                f"Short words, long words, and {target} all stay clear.",
                f"I keep my pace even while the sentence grows around {target}.",
            ]
        )
        prompt = build_generation_prompt(card)
        return GeneratedScriptDraft(
            title=title,
            body=body,
            prompt_snapshot=prompt,
            provider=self.name,
            auth_source="local",
        )

    def generate_ladder(
        self,
        cards: list[ImprovementCard],
        theme: str = "",
    ) -> GeneratedLadderDraft:
        prompt = build_ladder_generation_prompt(cards, theme)
        targets = [card.target_key for card in cards[:4] if card.target_key]
        if not targets:
            targets = ["clear endings", "steady breath", "even pace"]
        theme_title = theme.strip()[:60] or "Clarity"
        title = f"{theme_title} Ladder"
        levels = []
        for level in range(1, 6):
            density = [
                "slow and clean",
                "repeat and contrast",
                "shape the sentence",
                "carry the rhythm",
                "performance pass",
            ][level - 1]
            lines = []
            for target in targets:
                if level == 1:
                    lines.append(f"{target}. {target}, clear and calm.")
                elif level == 2:
                    lines.append(f"{target} starts steady; {target} ends clean.")
                elif level == 3:
                    lines.append(f"I keep {target} crisp while the sentence grows around it.")
                elif level == 4:
                    lines.append(f"With a steady breath, {target} stays clear before, during, and after the turn.")
                else:
                    lines.append(f"I carry {target} through a longer line without rushing, clipping, or fading the final sound.")
            body = "\n".join(lines[:8])
            levels.append(
                LadderLevelDraft(
                    level=level,
                    title=f"Level {level}: {density.title()}",
                    body=body,
                    focus=tuple(targets),
                )
            )
        return GeneratedLadderDraft(
            title=title,
            theme=theme.strip() or "Targeted clarity practice",
            levels=tuple(levels),
            prompt_snapshot=prompt,
            provider=self.name,
            auth_source="local",
        )


class OpenAIScriptProvider:
    name = "openai"

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        app_settings = _practice_settings()
        model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI script generation.") from exc

        prompt = build_generation_prompt(card)
        text, auth_source = self._generate_text(
            OpenAI,
            model,
            system=(
                "You are a precise speech coach. Generate concise, speakable "
                "practice material. Do not include analysis or markdown."
            ),
            user_prompt=prompt,
            max_output_tokens=700,
        )
        title, body = parse_generated_script(text, fallback_title=f"Drill: {card.title}")
        return GeneratedScriptDraft(
            title=title,
            body=body,
            prompt_snapshot=prompt,
            provider=f"openai:{model}",
            auth_source=auth_source,
        )

    def generate_ladder(
        self,
        cards: list[ImprovementCard],
        theme: str = "",
    ) -> GeneratedLadderDraft:
        app_settings = _practice_settings()
        model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI ladder generation.") from exc
        prompt = build_ladder_generation_prompt(cards, theme)
        text, auth_source = self._generate_text(
            OpenAI,
            model,
            system=(
                "You are a speech coach and drill designer. Return strict JSON only. "
                "Make the practice useful, targeted, and memorable."
            ),
            user_prompt=prompt,
            max_output_tokens=2200,
        )
        title, ladder_theme, levels = parse_generated_ladder(text, fallback_title="Generated Practice Ladder")
        return GeneratedLadderDraft(
            title=title,
            theme=ladder_theme,
            levels=tuple(levels),
            prompt_snapshot=prompt,
            provider=f"openai:{model}",
            auth_source=auth_source,
        )

    def _generate_text(
        self,
        openai_class,
        model: str,
        *,
        system: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> tuple[str, str]:
        app_settings = _practice_settings()
        api_key = (app_settings.get_secret("openai_api_key") if app_settings else None) or settings.OPENAI_API_KEY
        codex_token = _codex_access_token(app_settings)
        if codex_token:
            try:
                text = self._codex_response(
                    openai_class,
                    codex_token,
                    model,
                    system=system,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                )
                return text, "codex"
            except Exception:
                if not api_key:
                    raise
        if not api_key:
            raise RuntimeError("OpenAI generation requires a Codex login or an OpenAI API key.")
        client = openai_class(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=max_output_tokens,
        )
        return _response_text(response), "api_key" if not codex_token else "api_key_fallback"

    def _codex_response(
        self,
        openai_class,
        codex_token: str,
        model: str,
        *,
        system: str,
        user_prompt: str,
        max_output_tokens: int,
    ):
        client = openai_class(api_key=codex_token, base_url=settings.CODEX_CHATGPT_BASE_URL)
        kwargs = {
            "model": model,
            "instructions": system,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                }
            ],
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        fallback = dict(kwargs)
        for _attempt in range(3):
            try:
                request = dict(fallback)
                request["stream"] = True
                return _stream_response_text(client.responses.create(**request))
            except Exception as exc:
                msg = str(exc).lower()
                changed = False
                if "unsupported parameter: max_output_tokens" in msg and "max_output_tokens" in fallback:
                    fallback.pop("max_output_tokens", None)
                    changed = True
                if "unsupported parameter: store" in msg and "store" in fallback:
                    fallback.pop("store", None)
                    changed = True
                if not changed:
                    raise
        raise RuntimeError("Codex streaming response failed after parameter fallback.")


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
            auth_source="api_key",
        )

    def generate_ladder(
        self,
        cards: list[ImprovementCard],
        theme: str = "",
    ) -> GeneratedLadderDraft:
        app_settings = _practice_settings()
        api_key = (app_settings.get_secret("anthropic_api_key") if app_settings else None) or settings.ANTHROPIC_API_KEY
        model = (app_settings.anthropic_script_model if app_settings else None) or settings.ANTHROPIC_SCRIPT_MODEL
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic ladder generation.")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the optional 'anthropic' package to use Anthropic ladder generation.") from exc
        prompt = build_ladder_generation_prompt(cards, theme)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=2200,
            system="You are a speech coach and drill designer. Return strict JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        title, ladder_theme, levels = parse_generated_ladder(
            "\n".join(parts),
            fallback_title="Generated Practice Ladder",
        )
        return GeneratedLadderDraft(
            title=title,
            theme=ladder_theme,
            levels=tuple(levels),
            prompt_snapshot=prompt,
            provider=f"anthropic:{model}",
            auth_source="api_key",
        )


def generate_local_template(card: ImprovementCard) -> GeneratedScriptDraft:
    return LocalTemplateScriptProvider().generate(card)


def generate_script_draft(
    card: ImprovementCard,
    provider_name: str | None = None,
) -> GeneratedScriptDraft:
    return get_script_generation_provider(provider_name).generate(card)


def generate_ladder_draft(
    cards: list[ImprovementCard],
    theme: str = "",
    provider_name: str | None = None,
) -> GeneratedLadderDraft:
    provider = get_script_generation_provider(provider_name)
    return provider.generate_ladder(cards, theme=theme)


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
        ("openai", _openai_choice_label(app_settings, openai_model)),
        ("anthropic", f"Anthropic ({anthropic_model})"),
    ]


def _practice_settings() -> PracticeSettings | None:
    try:
        return PracticeSettings.load()
    except (OperationalError, ProgrammingError):
        return None


def _openai_choice_label(app_settings: PracticeSettings | None, model: str) -> str:
    if app_settings is None:
        return f"OpenAI ({model}, Codex/API key required)"
    has_codex = bool(deserialize_token_bundle(app_settings.get_secret("codex_token_bundle")))
    has_api_key = bool(app_settings.get_secret("openai_api_key") or settings.OPENAI_API_KEY)
    if has_codex and has_api_key:
        auth_label = "Codex auth with API key fallback"
    elif has_codex:
        auth_label = "Codex auth"
    elif has_api_key:
        auth_label = "API key"
    else:
        auth_label = "Codex/API key required"
    return f"OpenAI ({model}, {auth_label})"


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


def parse_generated_ladder(text: str, fallback_title: str) -> tuple[str, str, list[LadderLevelDraft]]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback_title, "Generated practice ladder", [
            LadderLevelDraft(level=1, title="Generated Drill", body=cleaned or "Read this line clearly.")
        ]
    title = str(raw.get("title") or fallback_title).strip()[:255]
    theme = str(raw.get("theme") or "").strip()
    levels = []
    for idx, row in enumerate(raw.get("levels") or [], start=1):
        if not isinstance(row, dict):
            continue
        try:
            level = int(row.get("level") or idx)
        except (TypeError, ValueError):
            level = idx
        level_title = str(row.get("title") or f"Level {level}").strip()[:255]
        lines = row.get("lines") or row.get("body") or []
        if isinstance(lines, str):
            body = lines.strip()
        else:
            body = "\n".join(str(line).strip() for line in lines if str(line).strip())
        focus_raw = row.get("focus") or []
        if isinstance(focus_raw, str):
            focus = (focus_raw,)
        else:
            focus = tuple(str(item).strip() for item in focus_raw if str(item).strip())
        if body:
            levels.append(LadderLevelDraft(level=level, title=level_title, body=body, focus=focus))
    return title or fallback_title, theme, levels or [
        LadderLevelDraft(level=1, title="Generated Drill", body="Read this line clearly.")
    ]


def _response_text(response) -> str:
    text = getattr(response, "output_text", "") or ""
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if hasattr(part, "text") and hasattr(part.text, "value"):
                chunks.append(str(part.text.value))
            elif hasattr(part, "text"):
                chunks.append(str(part.text))
    return "".join(chunks) or str(response)


def _stream_response_text(event_stream) -> str:
    chunks = []
    for event in event_stream:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if isinstance(delta, str) and delta:
                chunks.append(delta)
            continue
        if event_type == "response.completed":
            completed_text = _response_text(getattr(event, "response", None))
            if completed_text and not chunks:
                chunks.append(completed_text)
    return "".join(chunks).strip()


def _codex_access_token(app_settings: PracticeSettings | None) -> str | None:
    if app_settings is None:
        return None
    bundle = deserialize_token_bundle(app_settings.get_secret("codex_token_bundle"))
    if not bundle:
        return None
    if not access_token_expires_soon(bundle):
        return bundle.get("access_token")
    try:
        refreshed = refresh_token_bundle(bundle)
    except CodexAuthError:
        return bundle.get("access_token")
    if refreshed != bundle:
        app_settings.set_secret("codex_token_bundle", serialize_token_bundle(refreshed))
        app_settings.save(update_fields=["codex_token_bundle_enc", "updated_at"])
    return refreshed.get("access_token")
