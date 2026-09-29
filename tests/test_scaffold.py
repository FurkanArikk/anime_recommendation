"""Smoke tests for configuration, CLI wiring and the health endpoint."""

import pytest
from fastapi.testclient import TestClient

from anime_rec.api.main import app
from anime_rec.cli import build_parser
from anime_rec.config import Settings


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.cloud.qdrant.io:6333")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.qdrant_url == "https://example.cloud.qdrant.io:6333"
    assert s.embedding_dim == 1536
    # Secrets must not leak through repr/logging.
    assert "secret" not in repr(s)


def test_invalid_embedding_dim_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIM", "1000")
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("command", ["enrich", "ingest", "embed", "index", "serve", "eval"])
def test_cli_has_all_stages(command: str) -> None:
    args = build_parser().parse_args([command])
    assert args.command == command


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
