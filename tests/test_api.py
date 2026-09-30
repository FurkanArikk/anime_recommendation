"""API tests: the real FastAPI app with a fake service injected (no model, no cluster)."""

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anime_rec.api.main import create_app
from anime_rec.config import Settings
from anime_rec.recommender.explain import Explainer
from anime_rec.recommender.service import RecommenderService
from tests.fakes import make_service


class FakeGemini:
    """Picks the candidates in reverse order, like a re-ranker would."""

    def __init__(self) -> None:
        self.models = self
        self.prompts: list[str] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.prompts.append(contents)
        ids = [
            json.loads(line)["anime_id"] for line in contents.splitlines() if line.startswith("{")
        ]
        picks = [{"anime_id": i, "reason": f"reason for {i}"} for i in reversed(ids)]
        return SimpleNamespace(text=json.dumps({"picks": picks, "summary": "A fitting mix."}))


@pytest.fixture
def gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture
def service(gemini: FakeGemini) -> RecommenderService:
    settings = Settings(_env_file=None, gemini_chat_fallback_models=[])  # type: ignore[call-arg]
    return make_service(explainer=Explainer(settings, client=gemini))


@pytest.fixture
def client(service: RecommenderService) -> Iterator[TestClient]:
    with TestClient(create_app(lambda _settings: service)) as c:  # runs the lifespan
        yield c


def ids(body: dict[str, Any]) -> list[int]:
    return [item["anime_id"] for item in body["items"]]


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["points"] == 9
    assert ready["embedding_model"] == "keyword"


def test_openapi_docs_available(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/similar", "/recommend", "/titles", "/facets"} <= set(paths)


def test_get_anime_and_404(client: TestClient) -> None:
    assert client.get("/anime/5").json()["title"] == "Death Note"
    response = client.get("/anime/424242")
    assert response.status_code == 404 and "424242" in response.json()["detail"]


def test_titles_autocomplete(client: TestClient) -> None:
    titles = client.get("/titles", params={"q": "kyojin"}).json()
    assert [t["anime_id"] for t in titles][:1] == [1]  # most members first


def test_facets(client: TestClient) -> None:
    facets = client.get("/facets").json()
    assert {"value": "Action", "count": 4} in facets["genres"]
    assert facets["year_min"] == 2000 and facets["year_max"] == 2019


def test_search_with_filters(client: TestClient) -> None:
    body = client.post(
        "/search", json={"query": "detective", "filters": {"min_score": 8.7}, "limit": 1}
    ).json()
    assert ids(body) == [6] and not body["explained"] and body["took_ms"] >= 0


def test_search_with_explanation_reranks_and_is_grounded(
    client: TestClient, gemini: FakeGemini
) -> None:
    body = client.post("/search", json={"query": "titan wall", "limit": 2, "explain": True}).json()
    assert body["explained"] and body["summary"] == "A fitting mix."
    assert all(item["reason"] for item in body["items"])
    # The LLM saw 2x limit candidates and its (reversed) choice was honoured.
    prompt_ids = [json.loads(x)["anime_id"] for x in gemini.prompts[0].splitlines() if x[:1] == "{"]
    assert len(prompt_ids) == 4 and ids(body) == list(reversed(prompt_ids))[:2]


def test_similar_by_title(client: TestClient) -> None:
    body = client.post("/similar", json={"title": "shingeki no kyojin", "limit": 3}).json()
    assert body["seeds"][0]["anime_id"] == 1
    assert not {1, 2, 3} & set(ids(body))


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"limit": 3}, 422),  # neither anime_id nor title
        ({"anime_id": 1, "title": "x", "limit": 3}, 422),  # both
        ({"title": "no such anime at all", "limit": 3}, 404),
        ({"anime_id": 424242}, 404),
    ],
)
def test_similar_errors(client: TestClient, payload: dict[str, Any], status: int) -> None:
    assert client.post("/similar", json=payload).status_code == status


def test_recommend_taste_profile(client: TestClient) -> None:
    body = client.post("/recommend", json={"liked": [5], "disliked": [7], "limit": 3}).json()
    assert ids(body)[0] == 6
    assert [s["anime_id"] for s in body["seeds"]] == [5, 7]


@pytest.mark.parametrize(
    "payload",
    [
        {"liked": []},
        {"liked": [5], "limit": 999},
        {"liked": [5], "filters": {"year_min": 2020, "year_max": 2010}},
        {"liked": [5], "strategy": "magic"},
    ],
)
def test_recommend_validation(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/recommend", json=payload).status_code == 422


def test_empty_query_rejected(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_popular_and_title_thumbnails(client: TestClient) -> None:
    popular = client.get("/popular", params={"limit": 2}).json()
    assert [a["title"] for a in popular] == ["Death Note", "Shingeki no Kyojin"]  # by members
    titles = client.get("/titles", params={"q": "death"}).json()
    assert titles[0]["image_url"].startswith("https://cdn.myanimelist.net/")


def test_similar_points_to_the_seeds_other_seasons(client: TestClient) -> None:
    body = client.post("/similar", json={"anime_id": 1, "limit": 3}).json()
    assert not {1, 2, 3} & set(ids(body))  # sequels are not recommendations...
    (note,) = body["franchises"]  # ...but they are surfaced as a note
    assert note["seed"]["anime_id"] == 1
    assert [e["anime_id"] for e in note["entries"]] == [2, 3]
