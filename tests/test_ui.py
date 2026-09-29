"""Streamlit UI tests with AppTest: the real script runs against a fake API client."""

from pathlib import Path
from typing import Any, ClassVar

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from anime_rec.ui import api_client

# Resolved from the package so the test works from any working directory.
APP = str(Path(api_client.__file__).with_name("app.py"))


def item(anime_id: int, title: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "anime_id": anime_id,
        "title": title,
        "title_english": f"{title} EN",
        "synopsis": "Plot.",
        "type": "TV",
        "episodes": 12,
        "start_year": 2015,
        "season": "Fall 2015",
        "score": 8.1,
        "rank": 10,
        "members": 100_000,
        "image_url": "https://cdn.myanimelist.net/x.jpg",
        "mal_url": f"https://myanimelist.net/anime/{anime_id}",
        "genres": ["Drama"],
        "themes": [],
        "demographics": [],
        "studios": ["Madhouse"],
        "directors": [],
        "source": "Manga",
        "main_characters": [
            {"name": "Hero", "role": "Main", "image_url": None, "voice_actor": "VA"}
        ],
        "reason": reason,
    }


class FakeApi:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    down: ClassVar[bool] = False

    def __init__(self, base_url: str) -> None:
        pass

    def ready(self) -> dict[str, Any]:
        if FakeApi.down:
            raise api_client.ApiError("API unreachable: connection refused")
        return {"points": 9999, "embedding_model": "m", "chat_models": ["gemini-2.5-flash"]}

    def facets(self) -> dict[str, Any]:
        return {
            "genres": [{"value": "Drama", "count": 3}],
            "themes": [],
            "demographics": [],
            "type": [{"value": "TV", "count": 3}],
            "year_min": 1990,
            "year_max": 2025,
        }

    def titles(self, limit: int = 10_000) -> list[dict[str, Any]]:
        return [{"anime_id": 1535, "title": "Death Note", "title_english": None}]

    def chat(self, message: str, **payload: Any) -> dict[str, Any]:
        FakeApi.calls.append(("chat", {"message": message, **payload}))
        return {
            "items": [item(19, "Monster", reason="A patient cat-and-mouse thriller.")],
            "summary": "Tense mind games.",
            "explained": True,
            "took_ms": 1234,
            "understood": {
                "mode": "similar",
                "query": None,
                "unresolved_titles": [],
                "liked": [{"anime_id": 1535, "title": "Death Note"}],
                "disliked": [],
                "filters": {"max_episodes": 30},
            },
        }

    def similar(self, anime_id: int, **payload: Any) -> dict[str, Any]:
        FakeApi.calls.append(("similar", {"anime_id": anime_id, **payload}))
        return {
            "items": [item(33, "Psycho-Pass")],
            "summary": None,
            "explained": False,
            "took_ms": 80,
            "seeds": [{"anime_id": anime_id, "title": "Monster"}],
        }


@pytest.fixture(autouse=True)
def fake_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "ApiClient", FakeApi)
    FakeApi.calls, FakeApi.down = [], False
    st.cache_data.clear()
    st.cache_resource.clear()


def run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


def test_first_screen_shows_examples_and_filters() -> None:
    at = run()
    assert len(at.button) >= 4  # example prompts
    assert at.sidebar.multiselect[0].label == "👍 Liked"
    assert "9,999 anime" in markdown(at)


def test_chat_turn_renders_understood_cards_and_reason() -> None:
    at = run()
    at.chat_input[0].set_value("like death note, short").run()
    assert not at.exception
    text = markdown(at)
    assert "Monster" in text and "A patient cat-and-mouse thriller." in text
    assert "Similar to" in text and "Death Note" in text and "max_episodes=30" in text
    assert "Tense mind games." in text and "Fall 2015" in text
    kind, sent = FakeApi.calls[0]
    assert kind == "chat" and sent["explain"] is True and sent["liked"] == []


def test_sidebar_taste_is_sent_with_the_message() -> None:
    at = run()
    at.sidebar.multiselect[0].set_value(["Death Note"]).run()
    at.chat_input[0].set_value("something darker").run()
    assert FakeApi.calls[-1][1]["liked"] == [1535]


def test_more_like_this_button_calls_similar() -> None:
    at = run()
    at.chat_input[0].set_value("thriller").run()
    next(b for b in at.button if b.label == "More like this").click().run()
    assert not at.exception
    kind, sent = FakeApi.calls[-1]
    assert kind == "similar" and sent["anime_id"] == 19 and sent["limit"] == 6
    assert "Psycho-Pass" in markdown(at)


def test_api_down_shows_error_instead_of_crashing() -> None:
    FakeApi.down = True
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert "not available" in at.error[0].value
