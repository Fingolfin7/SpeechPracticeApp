from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from practice.models import PracticeScript


@dataclass(frozen=True)
class ScriptImportItem:
    title: str
    body: str
    author: str = ""
    tags: tuple[str, ...] = ()
    source_ref: str = ""


@dataclass(frozen=True)
class ScriptImportResult:
    created: int
    updated: int
    skipped: int
    items: tuple[PracticeScript, ...]


def import_script_items(
    items: Iterable[ScriptImportItem],
    *,
    source: str = PracticeScript.SOURCE_IMPORTED,
    extra_tags: Iterable[str] = (),
    replace: bool = True,
) -> ScriptImportResult:
    created = 0
    updated = 0
    skipped = 0
    imported: list[PracticeScript] = []
    common_tags = tuple(tag for tag in extra_tags if tag)

    for item in items:
        title = normalize_space(item.title)
        body = normalize_body(item.body)
        if not title or not body:
            skipped += 1
            continue

        tags = sorted({*item.tags, *common_tags})
        defaults = {
            "author": normalize_space(item.author),
            "body": body,
            "source": source,
            "source_ref": item.source_ref[:512],
            "tags": tags,
            "active": True,
        }
        lookup = {
            "title": title,
            "author": defaults["author"],
            "source_ref": defaults["source_ref"],
        }
        if replace:
            script, was_created = PracticeScript.objects.update_or_create(
                **lookup,
                defaults=defaults,
            )
        else:
            script = PracticeScript.objects.filter(**lookup).first()
            if script is not None:
                skipped += 1
                continue
            script = PracticeScript.objects.create(title=title, **defaults)
            was_created = True

        created += 1 if was_created else 0
        updated += 0 if was_created else 1
        imported.append(script)

    return ScriptImportResult(
        created=created,
        updated=updated,
        skipped=skipped,
        items=tuple(imported),
    )


def parse_script_upload(
    *,
    name: str,
    content: bytes,
    default_author: str = "",
) -> list[ScriptImportItem]:
    suffix = Path(name).suffix.lower()
    if suffix == ".zip":
        return parse_zip_bytes(name=name, content=content, default_author=default_author)
    if suffix == ".csv":
        return parse_csv_text(
            content.decode("utf-8-sig"),
            source_ref=name,
            default_author=default_author,
        )
    if suffix == ".json":
        return parse_json_text(
            content.decode("utf-8-sig"),
            source_ref=name,
            default_author=default_author,
        )
    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig")
        return [parse_plain_text(text, source_ref=name, default_author=default_author)]
    return []


def parse_path(path: Path, *, default_author: str = "") -> list[ScriptImportItem]:
    path = Path(path)
    if path.is_dir():
        items: list[ScriptImportItem] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".txt", ".md", ".csv", ".json", ".zip"}:
                items.extend(parse_path(child, default_author=default_author))
        return items
    return parse_script_upload(
        name=str(path),
        content=path.read_bytes(),
        default_author=default_author,
    )


def parse_zip_bytes(
    *,
    name: str,
    content: bytes,
    default_author: str = "",
) -> list[ScriptImportItem]:
    items: list[ScriptImportItem] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in {".txt", ".md", ".csv", ".json"}:
                continue
            nested = archive.read(info)
            items.extend(
                parse_script_upload(
                    name=f"{name}:{info.filename}",
                    content=nested,
                    default_author=default_author,
                )
            )
    return items


def parse_plain_text(
    text: str,
    *,
    source_ref: str,
    default_author: str = "",
) -> ScriptImportItem:
    title = Path(source_ref.split(":", 1)[-1]).stem
    body = text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("title:"):
        title = lines[0].split(":", 1)[1].strip() or title
        body = "\n".join(text.splitlines()[1:]).strip()
    author = default_author
    if lines and len(lines) > 1 and lines[1].lower().startswith("author:"):
        author = lines[1].split(":", 1)[1].strip()
        body = "\n".join(text.splitlines()[2:]).strip()
    return ScriptImportItem(
        title=title,
        author=author,
        body=body,
        tags=("upload",),
        source_ref=source_ref,
    )


def parse_json_text(
    text: str,
    *,
    source_ref: str,
    default_author: str = "",
) -> list[ScriptImportItem]:
    raw = json.loads(text)
    rows: list[dict | tuple[str, object]] = []
    if isinstance(raw, dict):
        rows = [(key, value) for key, value in raw.items()]
    elif isinstance(raw, list):
        rows = [row for row in raw if isinstance(row, dict)]

    items: list[ScriptImportItem] = []
    for idx, row in enumerate(rows):
        if isinstance(row, tuple):
            title, value = row
            if isinstance(value, dict):
                body = pick_field(value, "body", "poem", "content", "text")
                author = pick_field(value, "author", "poet", default=default_author)
                tags = split_tags(pick_field(value, "tags", "tag", default=""))
            else:
                body = str(value)
                author = default_author
                tags = ()
        else:
            title = pick_field(row, "title", "name", default=f"Script {idx + 1}")
            body = pick_field(row, "body", "poem", "content", "text")
            author = pick_field(row, "author", "poet", default=default_author)
            tags = split_tags(pick_field(row, "tags", "tag", default=""))
        items.append(
            ScriptImportItem(
                title=str(title),
                body=str(body),
                author=str(author),
                tags=tuple(tags),
                source_ref=f"{source_ref}#{idx}",
            )
        )
    return items


def parse_csv_text(
    text: str,
    *,
    source_ref: str,
    default_author: str = "",
) -> list[ScriptImportItem]:
    reader = csv.DictReader(io.StringIO(text))
    items: list[ScriptImportItem] = []
    for idx, row in enumerate(reader):
        normalized = {normalize_key(key): value for key, value in row.items() if key is not None}
        title = pick_field(normalized, "title", "name", default=f"Script {idx + 1}")
        body = pick_field(normalized, "poem", "body", "content", "text")
        author = pick_field(normalized, "poet", "author", default=default_author)
        tags = split_tags(pick_field(normalized, "tags", "tag", default=""))
        items.append(
            ScriptImportItem(
                title=title,
                body=body,
                author=author,
                tags=tuple(tags),
                source_ref=f"{source_ref}#{idx}",
            )
        )
    return items


def pick_field(data: dict, *names: str, default: str = "") -> str:
    for name in names:
        value = data.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def split_tags(value: str) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [normalize_space(str(item)) for item in value if normalize_space(str(item))]
    value = value.replace("|", ",").replace(";", ",")
    return [normalize_space(part) for part in value.split(",") if normalize_space(part)]


def normalize_key(value: str) -> str:
    return normalize_space(value).lower().replace(" ", "_")


def normalize_space(value: str) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def normalize_body(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
