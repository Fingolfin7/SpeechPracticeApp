from __future__ import annotations

import json
from typing import Any
from urllib import parse, request, error


class AutumnError(Exception):
    pass


def normalize_base_url(base_url: str | None) -> str:
    base = (base_url or "https://autumn-lg0b.onrender.com").strip()
    return base.rstrip("/")


class AutumnClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = normalize_base_url(base_url)
        self.api_key = (api_key or "").strip()

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            clean = {k: v for k, v in params.items() if v not in (None, "")}
            if clean:
                url = f"{url}?{parse.urlencode(clean, doseq=True)}"
        return url

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
        req = request.Request(
            self._url(path, params),
            data=body,
            headers=self._headers(),
            method=method.upper(),
        )
        try:
            with request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            msg = raw
            try:
                payload = json.loads(raw)
                msg = payload.get("error") or payload.get("detail") or raw
            except Exception:
                pass
            raise AutumnError(f"Autumn returned {exc.code}: {msg}") from exc
        except error.URLError as exc:
            raise AutumnError(f"Could not reach Autumn: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AutumnError("Autumn request timed out") from exc

        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AutumnError("Autumn returned a non-JSON response") from exc
        if isinstance(payload, dict):
            return payload
        raise AutumnError("Autumn returned an unexpected response")

    def authenticate(self, username: str, password: str) -> str:
        payload = self._request(
            "POST",
            "/get-auth-token/",
            {"username": username, "password": password},
        )
        token = str(payload.get("token") or "").strip()
        if not token:
            raise AutumnError("Autumn did not return an API token")
        self.api_key = token
        return token

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/api/me/")

    def list_projects(self) -> list[str]:
        payload = self._request("GET", "/api/projects/", params={"compact": "true"})
        projects = payload.get("projects", [])
        return [str(p) for p in projects if str(p).strip()]

    def list_subprojects(self, project: str) -> list[str]:
        payload = self._request(
            "GET",
            "/api/subprojects/",
            params={"project": project, "compact": "true"},
        )
        subs = payload.get("subprojects", [])
        return [str(s) for s in subs if str(s).strip()]

    def timer_status(self, session_id: int | None = None, project: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"compact": "true"}
        if session_id:
            params["session_id"] = session_id
        if project:
            params["project"] = project
        return self._request("GET", "/api/timer/status/", params=params)

    def start_timer(
        self,
        project: str,
        subprojects: list[str] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"project": project, "subprojects": subprojects or []}
        if note:
            data["note"] = note
        return self._request("POST", "/api/timer/start/", data)

    def stop_timer(
        self,
        session_id: int | None = None,
        project: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if session_id:
            data["session_id"] = session_id
        if project:
            data["project"] = project
        if note is not None:
            data["note"] = note
        return self._request("POST", "/api/timer/stop/", data)
