"""Thin HTTP client for the backend. The UI never imports the recommender directly: it is
a separate service and talks to the API like any other client would."""

from typing import Any

import httpx


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"API unreachable: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{response.status_code}: {detail}")
        return response.json()

    def ready(self) -> dict[str, Any]:
        result: dict[str, Any] = self._call("GET", "/ready")
        return result

    def facets(self) -> dict[str, Any]:
        result: dict[str, Any] = self._call("GET", "/facets")
        return result

    def titles(self, limit: int = 10_000) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._call("GET", "/titles", params={"limit": limit})
        return result

    def chat(self, message: str, **payload: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._call("POST", "/chat", json={"message": message, **payload})
        return result

    def similar(self, anime_id: int, **payload: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._call(
            "POST", "/similar", json={"anime_id": anime_id, **payload}
        )
        return result
