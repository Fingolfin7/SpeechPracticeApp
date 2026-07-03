from __future__ import annotations

import json
import os
import re
import time
import uuid
from html import unescape
from pathlib import Path

import requests


BASE_URL = os.environ.get("SPEECHPRACTICE_BASE_URL", "https://speechpractice-web.onrender.com").rstrip("/")
USERNAME = os.environ["SPEECHPRACTICE_USERNAME"]
PASSWORD = os.environ["SPEECHPRACTICE_PASSWORD"]
AUDIO_PATH = Path(os.environ["SPEECHPRACTICE_AUDIO_PATH"]).resolve()


def csrf_from(html: str) -> str:
    match = re.search(r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']', html)
    if not match:
        match = re.search(r'value=["\']([^"\']+)["\']\s+name=["\']csrfmiddlewaretoken["\']', html)
    if not match:
        raise RuntimeError("Could not find CSRF token in HTML response.")
    return unescape(match.group(1))


def first_selected_script_id(html: str) -> str:
    match = re.search(r'<option\s+value=["\'](\d+)["\'][^>]*selected', html)
    if match:
        return match.group(1)
    match = re.search(r'<option\s+value=["\'](\d+)["\']', html)
    if not match:
        raise RuntimeError("Could not find a script option on /practice/.")
    return match.group(1)


def main() -> int:
    if not AUDIO_PATH.is_file():
        raise RuntimeError(f"Audio file does not exist: {AUDIO_PATH}")

    session = requests.Session()
    session.headers.update({"User-Agent": "SpeechPractice live smoke test"})

    login_page = session.get(f"{BASE_URL}/accounts/login/", timeout=120)
    login_page.raise_for_status()
    csrf = csrf_from(login_page.text)
    login_response = session.post(
        f"{BASE_URL}/accounts/login/",
        data={
            "csrfmiddlewaretoken": csrf,
            "username": USERNAME,
            "password": PASSWORD,
            "next": "",
        },
        headers={"Referer": f"{BASE_URL}/accounts/login/"},
        allow_redirects=True,
        timeout=120,
    )
    login_response.raise_for_status()
    if "Sign in to SpeechPractice" in login_response.text:
        raise RuntimeError("Login did not succeed.")

    practice_page = session.get(f"{BASE_URL}/practice/", timeout=120)
    practice_page.raise_for_status()
    csrf = csrf_from(practice_page.text)
    script_id = first_selected_script_id(practice_page.text)

    with AUDIO_PATH.open("rb") as handle:
        response = session.post(
            f"{BASE_URL}/practice/",
            data={
                "csrfmiddlewaretoken": csrf,
                "mode": "script",
                "script": script_id,
                "provider": "openai",
            },
            files={
                "audio": (AUDIO_PATH.name, handle, "audio/webm" if AUDIO_PATH.suffix == ".webm" else "application/octet-stream"),
            },
            headers={
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
                "X-Idempotency-Key": str(uuid.uuid4()),
                "Referer": f"{BASE_URL}/practice/",
            },
            timeout=120,
        )
    payload = response.json()
    if response.status_code >= 400 or not payload.get("ok"):
        raise RuntimeError(f"Scoring submission failed: HTTP {response.status_code} {payload}")

    status_url = payload.get("status_url")
    if not status_url:
        raise RuntimeError(f"No status_url returned: {payload}")
    if status_url.startswith("/"):
        status_url = f"{BASE_URL}{status_url}"

    final_payload = payload
    for _attempt in range(40):
        time.sleep(3)
        status_response = session.get(
            status_url,
            headers={"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
            timeout=120,
        )
        status_response.raise_for_status()
        final_payload = status_response.json()
        if final_payload.get("is_done") or final_payload.get("is_failed"):
            break

    result = {
        "initial_status": payload.get("status"),
        "status_url": status_url,
        "final_status": final_payload.get("status"),
        "is_done": bool(final_payload.get("is_done")),
        "is_failed": bool(final_payload.get("is_failed")),
        "error_message": final_payload.get("error_message") or "",
        "score_text": final_payload.get("score_text") or "",
        "metrics": final_payload.get("metrics") or {},
        "partial_transcript_preview": (final_payload.get("partial_transcript") or "")[:180],
    }
    print(json.dumps(result, indent=2))
    return 0 if result["is_done"] and not result["is_failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
