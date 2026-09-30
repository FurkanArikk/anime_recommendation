"""Chat: intent validation, filter merging, routing, and the /chat endpoint (Gemini faked)."""

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anime_rec.api.main import create_app
from anime_rec.config import Settings
from anime_rec.recommender.chat import respond
from anime_rec.recommender.filters import SearchFilters
from anime_rec.recommender.intent import (
    Intent,
    IntentFilters,
    IntentParser,
    TitleMention,
    merge_filters,
    validated_filters,
)
from anime_rec.recommender.llm import GeminiJson
from anime_rec.recommender.service import RecommenderService
from tests.fakes import make_service


@pytest.fixture
def service() -> RecommenderService:
    return make_service()


def test_validated_filters_drop_invented_values(service: RecommenderService) -> None:
    raw = IntentFilters(
        include_tags=["suspense", "Cyberpunk-Noir"],  # case-insensitive; second doesn't exist
        types=["movie", "Hologram"],
        year_min=2030,
        year_max=2001,  # swapped
        min_score=42,
        max_episodes=0,
    )
    f = validated_filters(raw, service.facets)
    assert f.include_tags == ["Suspense"] and f.types == ["Movie"]
    assert (f.year_min, f.year_max) == (2001, 2030)
    assert f.min_score == 10 and f.max_episodes is None


def test_ui_filters_win_and_tags_union() -> None:
    ui = SearchFilters(min_score=8.0, include_tags=["Action"], types=["TV"])
    parsed = SearchFilters(min_score=6.0, max_episodes=12, include_tags=["Comedy"], types=["Movie"])
    merged = merge_filters(ui, parsed)
    assert merged.min_score == 8.0 and merged.max_episodes == 12
    assert merged.include_tags == ["Action", "Comedy"] and merged.types == ["TV"]


def intent(**kw: Any) -> Intent:
    return Intent(**kw)


def test_text_only_routes_to_search(service: RecommenderService) -> None:
    rec, u = respond(service, "boxing", intent=intent(query="boxing underdog"), explain=False)
    assert u.mode == "search" and rec.items[0].anime_id == 8


def test_single_title_routes_to_similar_via_mal_title(service: RecommenderService) -> None:
    # The user wrote the English name; the parser supplied the MAL title.
    i = intent(liked=[TitleMention(as_written="Attack on Titan", mal_title="Shingeki no Kyojin")])
    rec, u = respond(service, "anything like Attack on Titan?", intent=i, explain=False)
    assert u.mode == "similar" and [r.anime_id for r in u.liked] == [1]
    assert not {1, 2, 3} & {h.anime_id for h in rec.items}


def test_title_plus_text_routes_to_taste_with_modifier(service: RecommenderService) -> None:
    i = intent(query="detective mystery", liked=[TitleMention(as_written="death note")],
               disliked=[TitleMention(as_written="Yuru Camp")])  # fmt: skip
    rec, u = respond(
        service, "like death note, more detective, no camping", intent=i, explain=False
    )
    assert u.mode == "taste" and [r.title for r in u.disliked] == ["Yuru Camp"]
    assert rec.items[0].anime_id == 6
    assert service._embedder.queries == ["detective mystery"]  # type: ignore[attr-defined]


def test_unresolved_titles_reported(service: RecommenderService) -> None:
    i = intent(query="titan", liked=[TitleMention(as_written="Some Show That Does Not Exist")])
    _, u = respond(service, "q", intent=i, explain=False)
    assert u.unresolved_titles == ["Some Show That Does Not Exist"] and u.mode == "search"


def test_ui_picks_combine_with_message(service: RecommenderService) -> None:
    _, u = respond(service, "q", intent=intent(query="detective"), liked_ids=[5], explain=False)
    assert u.mode == "taste" and [r.anime_id for r in u.liked] == [5]


def test_without_parser_whole_message_is_searched(service: RecommenderService) -> None:
    rec, u = respond(service, "girls camp", intent=None, explain=False)
    assert u.mode == "search" and not u.parsed_by_llm and rec.items[0].anime_id == 7


class FakeGemini:
    """Returns a canned intent for the parser and reversed picks for the explainer."""

    def __init__(self, parsed: dict[str, Any]) -> None:
        self.models = self
        self.parsed = parsed
        self.systems: list[str] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.systems.append(config.system_instruction)
        if config.response_schema is Intent:
            assert "ALLOWED TAGS" in contents  # catalog values are offered to the model
            return SimpleNamespace(text=json.dumps(self.parsed))
        ids = [json.loads(x)["anime_id"] for x in contents.splitlines() if x.startswith("{")]
        picks = [{"anime_id": i, "reason": "fits"} for i in ids]
        return SimpleNamespace(text=json.dumps({"picks": picks, "summary": "ok"}))


@pytest.fixture
def client() -> Iterator[TestClient]:
    parsed = {
        "query": "psychological battle of wits",
        "liked": [{"as_written": "DN", "mal_title": "Death Note"}],
        "disliked": [],
        "filters": {"include_tags": ["Psychological"], "max_episodes": 100},
    }
    settings = Settings(_env_file=None, gemini_chat_fallback_models=[])  # type: ignore[call-arg]
    llm = GeminiJson(settings, client=FakeGemini(parsed))
    from anime_rec.recommender.explain import Explainer

    service = make_service(explainer=Explainer(llm=llm))
    service._intent_parser = IntentParser(llm)
    with TestClient(create_app(lambda _s: service)) as c:
        yield c


def test_chat_endpoint(client: TestClient) -> None:
    body = client.post("/chat", json={"message": "something like DN, psychological"}).json()
    u = body["understood"]
    assert u["mode"] == "taste" and u["parsed_by_llm"]
    assert u["liked"][0]["title"] == "Death Note"
    assert u["filters"]["include_tags"] == ["Psychological"]
    assert [i["anime_id"] for i in body["items"]] == [6]  # only Monster is psychological
    assert body["explained"] and body["items"][0]["reason"] == "fits"
    assert body["franchises"] == []  # Death Note has no other entries in the test catalog


def test_chat_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_liked_anime_with_sequels_gets_a_franchise_note(service: RecommenderService) -> None:
    i = intent(liked=[TitleMention(as_written="Shingeki no Kyojin")])
    rec, _ = respond(service, "I liked Attack on Titan", intent=i, explain=False)
    assert [n.seed.anime_id for n in rec.franchises] == [1]
    assert [e.anime_id for e in rec.franchises[0].entries] == [2, 3]
