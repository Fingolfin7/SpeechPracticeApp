from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Protocol

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError

from practice.models import ImprovementCard, PracticeSession, PracticeSettings
from practice.services import local_drills
from practice.services.codex_auth import (
    codex_access_token,
    deserialize_token_bundle,
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


@dataclass(frozen=True)
class GeneratedCardDraft:
    title: str
    kind: str
    target_key: str
    prompt: str
    stats: dict


@dataclass(frozen=True)
class GeneratedCardSetDraft:
    cards: tuple[GeneratedCardDraft, ...]
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


DRILL_SYSTEM_PROMPT = (
    "You write short scripts that speech learners read aloud into a recorder to "
    "practice problem words and sounds. You are half drill designer, half writer: "
    "lines must be effortless to say, dense with the practice target, and vivid "
    "enough that reading them five times in a row stays fun. Follow the output "
    "format exactly."
)

LADDER_SYSTEM_PROMPT = (
    "You design five-level read-aloud practice ladders for a speech-training app. "
    "You are half articulation coach, half adventure writer: practice targets stay "
    "dense and deliberate, but the material has character and momentum. Return "
    "strict JSON only."
)


def _drill_target_description(card: ImprovementCard) -> str:
    target = card.target_key
    if card.kind == ImprovementCard.KIND_WORD:
        return f'the word "{target}"'
    if card.kind in (ImprovementCard.KIND_SOUND, ImprovementCard.KIND_CHARACTER):
        return f'the "{target}" sound'
    if card.kind == ImprovementCard.KIND_PHRASE:
        return f'the phrase "{target}"'
    if card.kind == ImprovementCard.KIND_POSITION:
        return f"sounds at the {target} of words"
    if card.kind == ImprovementCard.KIND_FLUENCY:
        return f"{target} (a pacing and fluency habit)"
    return f'"{target}"'


def _format_card_evidence(card: ImprovementCard) -> str:
    stats = card.stats or {}
    if stats.get("source") == "self_review":
        session_name = stats.get("source_session_name") or "a recent session"
        return f"the learner flagged this themselves while reviewing {session_name}"
    try:
        attempts = int(float(stats.get("attempts", 0)))
        errors = int(float(stats.get("errors", 0)))
    except (TypeError, ValueError):
        attempts = errors = 0
    try:
        rate = float(stats.get("error_rate", 0.0))
    except (TypeError, ValueError):
        rate = 0.0
    window = str(stats.get("source_window_label") or "").strip()
    if attempts > 0:
        evidence = f"the recognizer missed it in {errors} of {attempts} attempts ({rate:.0%} error rate)"
    elif rate > 0:
        evidence = f"recent takes show a {rate:.0%} error rate"
    else:
        evidence = "recent takes show recurring trouble with it"
    if window:
        evidence += f" over the {window}"
    return evidence


_DRILL_STRUCTURE_DEFAULT = (
    "- Lines 1-2: short sentences that exercise the target in an easy position.\n"
    "- Lines 3-5: medium sentences that stress the target in varied positions and "
    "put trickier material next to it.\n"
    "- Lines 6-8: longer lines with tighter consonant clusters and longer breath "
    "groups - harder, but still natural spoken English."
)

_DRILL_STRUCTURE_HINTS = {
    ImprovementCard.KIND_WORD: (
        "- Lines 1-2: short sentences with the target word in an easy, stressed position.\n"
        "- Lines 3-5: medium sentences that move the target to the start, middle, and end, "
        "and place trickier sounds next to it.\n"
        "- Lines 6-8: longer lines with tighter consonant clusters and longer breath "
        "groups - harder, but still natural spoken English."
    ),
    ImprovementCard.KIND_SOUND: (
        "- Lines 1-2: short sentences with the target sound once or twice in easy positions.\n"
        "- Lines 3-5: medium sentences that place the sound at word starts, middles, and ends, "
        "next to its nearest confusable sounds.\n"
        "- Lines 6-8: longer lines where the sound recurs in clusters and across breath "
        "groups - harder, but still natural spoken English."
    ),
    ImprovementCard.KIND_PHRASE: (
        "- Lines 1-2: the phrase inside very short sentences.\n"
        "- Lines 3-5: medium sentences that move the phrase to the start, middle, and end.\n"
        "- Lines 6-8: longer lines where the phrase must survive changing rhythm and a "
        "longer breath group."
    ),
    ImprovementCard.KIND_POSITION: (
        "- Lines 1-2: short sentences ending or starting on crisp, simple words.\n"
        "- Lines 3-5: medium sentences that stack several words with demanding sounds in "
        "the target position.\n"
        "- Lines 6-8: longer lines where the target position lands mid-breath and at the "
        "very end of the line."
    ),
    ImprovementCard.KIND_FLUENCY: (
        "- Lines 1-2: short lines with a natural pause built in.\n"
        "- Lines 3-5: medium sentences whose punctuation and clause breaks force pacing "
        "decisions.\n"
        "- Lines 6-8: longer lines with lists, asides, and turns that reward an even pace "
        "and controlled breath."
    ),
}
_DRILL_STRUCTURE_HINTS[ImprovementCard.KIND_CHARACTER] = _DRILL_STRUCTURE_HINTS[
    ImprovementCard.KIND_SOUND
]


def build_generation_prompt(card: ImprovementCard) -> str:
    target_desc = _drill_target_description(card)
    structure = _DRILL_STRUCTURE_HINTS.get(card.kind, _DRILL_STRUCTURE_DEFAULT)
    learner_note = ""
    if (card.stats or {}).get("source") == "self_review" and card.prompt:
        learner_note = f"The learner's own note about it: {card.prompt}\n"
    return (
        "Write an 8-line read-aloud drill.\n\n"
        f"Practice target: {target_desc}. Recent scoring: {_format_card_evidence(card)}.\n"
        f"{learner_note}"
        "\nStructure (do not label the lines):\n"
        f"{structure}\n"
        "\nWriting rules:\n"
        "- Every line is a concrete sentence you can picture: give it people, places, "
        "objects, small stakes. A drill can be a tiny scene.\n"
        "- Every line must put the practice target to work; for word, sound, and phrase "
        "targets that means the target appears in the line, and twice is fine when it flows.\n"
        "- Never reuse a sentence frame or an opening word across lines.\n"
        '- Banned: coaching filler ("clear and calm", "steady breath", "I say", "I practice"), '
        "sentences about speaking or practicing itself, and nonsense tongue-twisters that are "
        "not real English.\n"
        "- Mix line lengths so the reader has somewhere to breathe.\n\n"
        "Return exactly this format and nothing else:\n"
        "TITLE: <short, evocative title>\n"
        "SCRIPT:\n"
        "<line 1>\n"
        "<line 2>\n"
        "...\n"
        "<line 8>"
    )


def build_ladder_generation_prompt(cards: list[ImprovementCard], theme: str = "") -> str:
    target_lines = []
    for card in cards:
        desc = _drill_target_description(card)
        desc = desc[0].upper() + desc[1:]
        target_lines.append(f"- {desc}: {_format_card_evidence(card)}.")
    targets_block = (
        "\n".join(target_lines)
        if target_lines
        else "- No specific targets were provided; build a general clarity ladder."
    )
    theme_line = theme.strip() or "none - invent a vivid setting yourself and commit to it"
    return (
        "Design a five-level read-aloud practice ladder.\n\n"
        "The learner records themselves reading each level aloud; software scores which "
        "words and sounds came out unclear. Levels must climb steadily in difficulty and "
        "keep the practice targets dense the whole way up.\n\n"
        "Practice targets, from the learner's recent scoring data:\n"
        f"{targets_block}\n\n"
        f"Creative theme requested by the learner: {theme_line}\n\n"
        "Build it like one escalating story, not five disconnected worksheets:\n"
        "1. Warm-up - short lines, targets in easy stressed positions, set the scene.\n"
        "2. Contrast pairs - targets against their nearest confusable sounds, still short.\n"
        "3. Density - full sentences with targets at the start, middle, and end of the line.\n"
        "4. Flow - paragraph-feel lines with real rhythm and breath turns; the scene should "
        "be going somewhere.\n"
        "5. Performance - one climactic passage that is genuinely hard but still natural "
        "spoken English.\n\n"
        "Writing rules:\n"
        "- Every line is concrete and picturable: use the theme's people, objects, and stakes.\n"
        "- Weave every practice target into every level; do not quarantine one target per level.\n"
        "- Never reuse a sentence frame across lines or levels.\n"
        '- Banned: coaching filler ("clear and calm", "steady breath"), lines about speaking '
        "or practicing itself, and nonsense strings that are not real English.\n"
        "- Each line must be readable aloud in one or two breaths.\n\n"
        "Return strict JSON only. No markdown, comments, or prose outside JSON. Schema:\n"
        "{\n"
        '  "title": "short ladder title",\n'
        '  "theme": "one sentence tying the targets to the creative theme",\n'
        '  "levels": [\n'
        '    {"level": 1, "title": "short level title", "focus": ["target"], "lines": ["line 1", "line 2", "..."]}\n'
        "  ]\n"
        "}\n"
        "Each level must contain 4 to 8 lines."
    )


def build_self_review_card_prompt(session: PracticeSession, notes: str) -> str:
    transcript = (session.transcript or "").strip()
    return (
        "Turn a learner's self-review notes into SpeechPractice improvement cards.\n"
        "The learner knows what they meant to say and has written where they noticed mistakes. "
        "Prefer concrete, reusable focus areas over one-off commentary.\n\n"
        "Available card kinds:\n"
        "- word: a specific word to practice\n"
        "- sound: an articulation or sound pattern, including final consonants or swallowed endings\n"
        "- phrase: a phrase the learner clipped, blurred, or rushed\n"
        "- position: a word-position issue such as final sounds or phrase endings\n"
        "- fluency: pace, pausing, rushing, breath, filler, or rhythm issues\n\n"
        f"Session: {session.script_name} at {session.timestamp}\n"
        f"Transcript:\n{transcript[:3000] or '[no transcript]'}\n\n"
        f"Self-review notes:\n{notes.strip()[:3000]}\n\n"
        "Return strict JSON only. Schema:\n"
        "{\n"
        '  "cards": [\n'
        '    {"kind": "word|sound|phrase|position|fluency", "target": "short target key", '
        '"title": "short display title", "prompt": "why and how to practice it"}\n'
        "  ]\n"
        "}\n"
        "In each card's prompt, quote or closely paraphrase the learner's own words and name "
        "the concrete failure (what happened, and where in the take) - this text later feeds "
        "drill generation, so specifics beat generic advice.\n"
        "Return 1 to 6 cards. Do not include cards for issues the notes do not mention."
    )


class LocalTemplateScriptProvider:
    name = "local_template"

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        body = "\n".join(local_drills.drill_lines(card.kind, card.target_key))
        return GeneratedScriptDraft(
            title=f"Drill: {card.display_title}",
            body=body,
            prompt_snapshot=build_generation_prompt(card),
            provider=self.name,
            auth_source="local",
        )

    def generate_ladder(
        self,
        cards: list[ImprovementCard],
        theme: str = "",
    ) -> GeneratedLadderDraft:
        prompt = build_ladder_generation_prompt(cards, theme)
        pairs = [(card.kind, card.target_key) for card in cards[:4] if card.target_key]
        focus = tuple(target for _kind, target in pairs) or ("clarity", "pacing")
        theme_title = theme.strip()[:60] or "Clarity"
        rng = random.Random()
        pools: dict[int, list[str]] = {}
        levels = []
        for level in range(1, 6):
            lines = local_drills.ladder_level_lines(pairs, level, rng, pools)
            levels.append(
                LadderLevelDraft(
                    level=level,
                    title=f"Level {level}: {local_drills.LEVEL_NAMES[level - 1]}",
                    body="\n".join(lines[:8]),
                    focus=focus,
                )
            )
        return GeneratedLadderDraft(
            title=f"{theme_title} Ladder",
            theme=theme.strip() or "Targeted clarity practice",
            levels=tuple(levels),
            prompt_snapshot=prompt,
            provider=self.name,
            auth_source="local",
        )


class OpenAIScriptProvider:
    name = "openai"

    def __init__(self, user=None) -> None:
        self.user = user

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        app_settings = _practice_settings(self.user or card.user)
        model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI script generation.") from exc

        prompt = build_generation_prompt(card)
        text, auth_source = self._generate_text(
            OpenAI,
            model,
            system=DRILL_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_output_tokens=700,
        )
        title, body = parse_generated_script(text, fallback_title=f"Drill: {card.display_title}")
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
        app_settings = _practice_settings(self.user or (cards[0].user if cards else None))
        model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI ladder generation.") from exc
        prompt = build_ladder_generation_prompt(cards, theme)
        text, auth_source = self._generate_text(
            OpenAI,
            model,
            system=LADDER_SYSTEM_PROMPT,
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
        app_settings = _practice_settings(self.user)
        api_key = (app_settings.get_secret("openai_api_key") if app_settings else None) or settings.OPENAI_API_KEY
        codex_token = codex_access_token(app_settings)
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
            if app_settings and (
                app_settings.has_secret("codex_token_bundle")
                or app_settings.has_secret("openai_api_key")
            ):
                raise RuntimeError(
                    "Stored OpenAI/Codex credentials could not be decrypted "
                    "(DJANGO_SECRET_KEY likely changed since they were saved). "
                    "Re-enter them on the Account page, or run "
                    "'manage.py reencrypt_secrets' with the previous key."
                )
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

    def __init__(self, user=None) -> None:
        self.user = user

    def generate(self, card: ImprovementCard) -> GeneratedScriptDraft:
        app_settings = _practice_settings(self.user or card.user)
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
            system=DRILL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        title, body = parse_generated_script("\n".join(parts), fallback_title=f"Drill: {card.display_title}")
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
        app_settings = _practice_settings(self.user or (cards[0].user if cards else None))
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
            system=LADDER_SYSTEM_PROMPT,
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
    return get_script_generation_provider(provider_name, user=card.user).generate(card)


def generate_ladder_draft(
    cards: list[ImprovementCard],
    theme: str = "",
    provider_name: str | None = None,
) -> GeneratedLadderDraft:
    user = cards[0].user if cards else None
    provider = get_script_generation_provider(provider_name, user=user)
    return provider.generate_ladder(cards, theme=theme)


def generate_cards_from_self_review(
    session: PracticeSession,
    notes: str,
    provider_name: str | None = None,
) -> GeneratedCardSetDraft:
    clean_notes = (notes or "").strip()
    prompt = build_self_review_card_prompt(session, clean_notes)
    app_settings = _practice_settings(session.user)
    provider = (
        provider_name
        or (app_settings.script_generation_provider if app_settings else None)
        or settings.SCRIPT_GENERATION_PROVIDER
    )
    if provider == "local_template":
        return GeneratedCardSetDraft(
            cards=tuple(_local_cards_from_self_review(session, clean_notes)),
            prompt_snapshot=prompt,
            provider="local_template",
            auth_source="local",
        )
    if provider == "openai":
        model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI card generation.") from exc
        text, auth_source = OpenAIScriptProvider(user=session.user)._generate_text(
            OpenAI,
            model,
            system=(
                "You are a precise speech coach. Extract only actionable, reusable "
                "practice cards from the learner's own self-review notes. Return JSON only."
            ),
            user_prompt=prompt,
            max_output_tokens=900,
        )
        cards = parse_self_review_cards(text, session=session)
        return GeneratedCardSetDraft(
            cards=tuple(cards),
            prompt_snapshot=prompt,
            provider=f"openai:{model}",
            auth_source=auth_source,
        )
    if provider == "anthropic":
        model = (app_settings.anthropic_script_model if app_settings else None) or settings.ANTHROPIC_SCRIPT_MODEL
        api_key = (app_settings.get_secret("anthropic_api_key") if app_settings else None) or settings.ANTHROPIC_API_KEY
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic card generation.")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the optional 'anthropic' package to use Anthropic card generation.") from exc
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=900,
            system="You are a precise speech coach. Return strict JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        cards = parse_self_review_cards("\n".join(parts), session=session)
        return GeneratedCardSetDraft(
            cards=tuple(cards),
            prompt_snapshot=prompt,
            provider=f"anthropic:{model}",
            auth_source="api_key",
        )
    raise ValueError(f"Unknown script generation provider: {provider}")


def get_script_generation_provider(provider_name: str | None = None, user=None) -> ScriptGenerationProvider:
    app_settings = _practice_settings(user)
    provider = provider_name or (app_settings.script_generation_provider if app_settings else None) or settings.SCRIPT_GENERATION_PROVIDER
    if provider == "local_template":
        return LocalTemplateScriptProvider()
    if provider == "openai":
        return OpenAIScriptProvider(user=user)
    if provider == "anthropic":
        return AnthropicScriptProvider(user=user)
    raise ValueError(f"Unknown script generation provider: {provider}")


def script_generation_provider_choices(user=None) -> list[tuple[str, str]]:
    app_settings = _practice_settings(user)
    openai_model = (app_settings.openai_script_model if app_settings else None) or settings.OPENAI_SCRIPT_MODEL
    anthropic_model = (app_settings.anthropic_script_model if app_settings else None) or settings.ANTHROPIC_SCRIPT_MODEL
    return [
        ("local_template", "Local template"),
        ("openai", _openai_choice_label(app_settings, openai_model)),
        ("anthropic", f"Anthropic ({anthropic_model})"),
    ]


def _practice_settings(user=None) -> PracticeSettings | None:
    try:
        return PracticeSettings.load(user)
    except (OperationalError, ProgrammingError, ValueError):
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


def parse_self_review_cards(
    text: str,
    *,
    session: PracticeSession,
) -> list[GeneratedCardDraft]:
    cleaned = _strip_json_fence(text)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        return _local_cards_from_self_review(session, cleaned)
    rows = raw.get("cards") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return []
    cards = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = _normalize_card_kind(row.get("kind"))
        target = _clean_target(row.get("target") or row.get("target_key") or row.get("focus"))
        if not kind or not target:
            continue
        key = (kind, target.lower())
        if key in seen:
            continue
        seen.add(key)
        title = _card_title(kind, str(row.get("title") or "").strip() or target)
        prompt = str(row.get("prompt") or row.get("reason") or "").strip()
        if not prompt:
            prompt = f"Self-review from {session.script_name}: practice {target}."
        cards.append(
            GeneratedCardDraft(
                title=title,
                kind=kind,
                target_key=target,
                prompt=prompt[:1200],
                stats=_self_review_stats(session),
            )
        )
        if len(cards) >= 6:
            break
    return cards


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


def _local_cards_from_self_review(
    session: PracticeSession,
    notes: str,
) -> list[GeneratedCardDraft]:
    chunks = [
        re.sub(r"^\s*[-*\u2022\d.)]+\s*", "", part).strip()
        for part in re.split(r"[\n;]+", notes or "")
    ]
    cards: list[GeneratedCardDraft] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk:
            continue
        kind = _kind_from_note(chunk)
        target = _target_from_note(chunk, kind)
        key = (kind, target.lower())
        if key in seen:
            continue
        seen.add(key)
        cards.append(
            GeneratedCardDraft(
                title=_card_title(kind, target),
                kind=kind,
                target_key=target,
                prompt=f"Self-review note from {session.script_name}: {chunk}",
                stats=_self_review_stats(session),
            )
        )
        if len(cards) >= 6:
            break
    return cards


def _strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _kind_from_note(note: str) -> str:
    lowered = note.lower()
    if any(token in lowered for token in ("rush", "too fast", "pace", "pause", "breath", "filler")):
        return ImprovementCard.KIND_FLUENCY
    if any(token in lowered for token in ("final consonant", "ending", "endings", "swallow", "mumble", "slur", "pronounc")):
        return ImprovementCard.KIND_SOUND
    quoted = re.search(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`', note)
    if quoted and len((quoted.group(1) or quoted.group(2) or quoted.group(3) or "").split()) > 1:
        return ImprovementCard.KIND_PHRASE
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", note)
    if len(words) <= 4:
        return ImprovementCard.KIND_WORD
    return ImprovementCard.KIND_PHRASE


def _target_from_note(note: str, kind: str) -> str:
    quoted = re.search(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`', note)
    if quoted:
        return _clean_target(quoted.group(1) or quoted.group(2) or quoted.group(3))
    lowered = note.lower()
    if kind == ImprovementCard.KIND_FLUENCY:
        if "pause" in lowered:
            return "pausing"
        if "breath" in lowered:
            return "breath control"
        return "rushing"
    if kind == ImprovementCard.KIND_SOUND:
        if "final consonant" in lowered or "ending" in lowered or "endings" in lowered:
            return "final consonants"
        if "mumble" in lowered:
            return "mumbling"
        if "swallow" in lowered:
            return "swallowed words"
    match = re.search(r"\b(?:on|word|phrase|saying|said)\s+([A-Za-z][A-Za-z' -]{1,80})", note, re.IGNORECASE)
    if match:
        return _clean_target(match.group(1))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", note)
    if words:
        return _clean_target(" ".join(words[: min(5, len(words))]))
    return _clean_target(note)


def _normalize_card_kind(value) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "words": ImprovementCard.KIND_WORD,
        "pronunciation": ImprovementCard.KIND_SOUND,
        "sound_pattern": ImprovementCard.KIND_SOUND,
        "sounds": ImprovementCard.KIND_SOUND,
        "phrases": ImprovementCard.KIND_PHRASE,
        "pacing": ImprovementCard.KIND_FLUENCY,
        "pace": ImprovementCard.KIND_FLUENCY,
        "rhythm": ImprovementCard.KIND_FLUENCY,
    }
    raw = aliases.get(raw, raw)
    valid = {value for value, _label in ImprovementCard.KIND_CHOICES}
    return raw if raw in valid else ""


def _clean_target(value) -> str:
    target = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;\"'`")
    return target[:255] or "self-review focus"


def _card_title(kind: str, target: str) -> str:
    labels = {
        ImprovementCard.KIND_WORD: "Word focus",
        ImprovementCard.KIND_SOUND: "Sound pattern",
        ImprovementCard.KIND_CHARACTER: "Character focus",
        ImprovementCard.KIND_POSITION: "Word position",
        ImprovementCard.KIND_PHRASE: "Phrase focus",
        ImprovementCard.KIND_FLUENCY: "Fluency focus",
    }
    return f"{labels.get(kind, 'Focus')}: {target}"[:255]


def _self_review_stats(session: PracticeSession) -> dict:
    return {
        "source": "self_review",
        "source_session_id": session.pk,
        "source_session_name": session.script_name,
        "source_session_timestamp": session.timestamp,
    }
