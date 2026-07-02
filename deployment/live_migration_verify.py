from __future__ import annotations

import json
import os
import re
from html import unescape

import requests


BASE_URL = os.environ.get("SPEECHPRACTICE_BASE_URL", "https://speechpractice-web.onrender.com").rstrip("/")
USERNAME = os.environ["SPEECHPRACTICE_USERNAME"]
PASSWORD = os.environ["SPEECHPRACTICE_PASSWORD"]


def csrf_from(html: str) -> str:
    match = re.search(r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']', html)
    if not match:
        match = re.search(r'value=["\']([^"\']+)["\']\s+name=["\']csrfmiddlewaretoken["\']', html)
    if not match:
        raise RuntimeError("Could not find CSRF token.")
    return unescape(match.group(1))


def login() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "SpeechPractice migration verifier"})
    page = session.get(f"{BASE_URL}/accounts/login/", timeout=120)
    page.raise_for_status()
    response = session.post(
        f"{BASE_URL}/accounts/login/",
        data={
            "csrfmiddlewaretoken": csrf_from(page.text),
            "username": USERNAME,
            "password": PASSWORD,
            "next": "",
        },
        headers={"Referer": f"{BASE_URL}/accounts/login/"},
        allow_redirects=True,
        timeout=120,
    )
    response.raise_for_status()
    if "Sign in to SpeechPractice" in response.text:
        raise RuntimeError("Login did not succeed.")
    return session


def main() -> int:
    session = login()
    sessions_page = session.get(f"{BASE_URL}/sessions/", timeout=120)
    sessions_page.raise_for_status()
    session_ids = sorted({int(match) for match in re.findall(r"/sessions/(\d+)/", sessions_page.text)})
    legacy_ids = [session_id for session_id in session_ids if session_id <= 69]
    if not legacy_ids:
        raise RuntimeError(f"No legacy session ids found in sessions page. Found: {session_ids[:12]}")

    audio_checks = []
    for session_id in legacy_ids[:8]:
        audio = session.get(
            f"{BASE_URL}/sessions/{session_id}/audio/",
            headers={"Range": "bytes=0-127"},
            allow_redirects=False,
            timeout=120,
        )
        audio_checks.append(
            {
                "session_id": session_id,
                "status_code": audio.status_code,
                "content_range": audio.headers.get("Content-Range", ""),
                "content_length": audio.headers.get("Content-Length", ""),
            }
        )
        if audio.status_code == 206:
            break

    result = {
        "listed_session_count": len(session_ids),
        "legacy_session_count_in_page": len(legacy_ids),
        "first_legacy_ids": legacy_ids[:8],
        "audio_checks": audio_checks,
        "range_audio_ok": any(check["status_code"] == 206 for check in audio_checks),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["range_audio_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
