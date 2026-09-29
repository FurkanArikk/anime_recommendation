"""Jikan enrichment tests: HTTP mocked with httpx.MockTransport (no network)."""

from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from anime_rec.config import Settings
from anime_rec.ingestion.clean import CleaningReport, RawTables, clean_anime, merge_enrichment
from anime_rec.ingestion.jikan import (
    JikanCache,
    JikanClient,
    JikanUnavailableError,
    RetryableJikanError,
    load_enrichment,
    parse_jikan,
    run_enrich,
)
from anime_rec.ingestion.schema import CLEAN_ANIME
from anime_rec.ratelimit import Throttle
from anime_rec.recommender.titles import TitleEntry, TitleIndex

DEATH_NOTE = {
    "mal_id": 1535,
    "title": "Death Note",
    "title_english": "Death Note",
    "title_japanese": "デスノート",
    "titles": [
        {"type": "Default", "title": "Death Note"},
        {"type": "Synonym", "title": "DN"},
        {"type": "Japanese", "title": "デスノート"},
        {"type": "English", "title": "Death Note"},
    ],
    "genres": [{"name": "Supernatural"}, {"name": "Suspense"}],
    "explicit_genres": [],
    "themes": [{"name": "Psychological"}],
    "demographics": [{"name": "Shounen"}],
    "source": "Manga",
    "rating": "R - 17+ (violence & profanity)",
    "season": "fall",
    "year": 2006,
}


def test_parse_jikan() -> None:
    parsed = parse_jikan(DEATH_NOTE)
    assert parsed["title_synonyms"] == ["DN"]  # default/Japanese/duplicate English dropped
    assert parsed["themes"] == ["Psychological"]
    assert parsed["season"] == "Fall 2006"
    assert parsed["source"] == "Manga"


def client_for(responses: dict[int, list[httpx.Response]]) -> tuple[JikanClient, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        anime_id = int(request.url.path.rsplit("/", 1)[1])
        calls.append(anime_id)
        return responses[anime_id].pop(0)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    no_wait = Throttle(6000, sleep=lambda _: None)
    return JikanClient(http, no_wait, max_attempts=3, retry_sleep=lambda _: None), calls


def ok(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"data": data})


def test_retries_504_then_succeeds() -> None:
    client, calls = client_for({1535: [httpx.Response(504), ok(DEATH_NOTE)]})
    assert client.fetch(1535)["mal_id"] == 1535
    assert calls == [1535, 1535]


def test_404_is_cached_as_not_found() -> None:
    client, _ = client_for({2: [httpx.Response(404)]})
    assert client.fetch(2) == {"_not_found": True}


def test_gives_up_after_max_attempts() -> None:
    client, calls = client_for({7: [httpx.Response(504)] * 3})
    with pytest.raises(RetryableJikanError):
        client.fetch(7)
    assert len(calls) == 3


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"anime_id": [1535, 2, 7], "rank": [2, 1, 3]}).to_csv(raw / "anime.csv")
    return Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]


def test_run_enrich_caches_and_resumes(settings: Settings) -> None:
    client, calls = client_for(
        {1535: [ok(DEATH_NOTE)], 2: [httpx.Response(404)], 7: [httpx.Response(504)] * 3}
    )
    stats = run_enrich(settings, client=client)
    assert stats == {"fetched": 1, "not_found": 1, "failed": 1}
    assert calls[:2] == [2, 1535]  # rank order

    # Second run only retries the failure.
    client2, calls2 = client_for({7: [ok({**DEATH_NOTE, "mal_id": 7, "title": "Seven"})]})
    run_enrich(settings, client=client2)
    assert calls2 == [7]
    enrichment = load_enrichment(settings.jikan_dir)
    assert enrichment is not None and sorted(enrichment.index) == [7, 1535]  # 404 excluded


def test_merge_fills_genres_and_keeps_schema(raw_tables: RawTables, tmp_path: Path) -> None:
    cache = JikanCache(tmp_path / "jikan")
    # One Piece (21) has scraped genres; add a Jikan genre. 99 had none at all.
    cache.put(21, {**DEATH_NOTE, "mal_id": 21, "title": "One Piece", "title_english": "One Piece",
                   "genres": [{"name": "Adventure"}], "themes": []})  # fmt: skip
    cache.put(99, {**DEATH_NOTE, "mal_id": 99, "title_english": "Mystery Film",
                   "genres": [{"name": "Mystery"}]})  # fmt: skip
    report = CleaningReport()
    out = clean_anime(raw_tables, report, load_enrichment(tmp_path / "jikan"))
    CLEAN_ANIME.validate(out)

    rows = out.set_index("anime_id")
    assert rows.loc[21, "genres"] == ["Action", "Adventure", "Slice of Life"]  # union
    assert rows.loc[99, "genres"] == ["Mystery"]
    assert rows.loc[99, "title_english"] == "Mystery Film"
    assert rows.loc[1535, "title_english"] is None  # not enriched: columns still present
    assert rows.loc[1535, "title_synonyms"] == []
    assert report.counts["main_genre_filled_by_jikan"] == 1


def test_merge_without_enrichment_adds_empty_columns(raw_tables: RawTables) -> None:
    out = clean_anime(raw_tables)
    CLEAN_ANIME.validate(out)
    assert out["title_english"].isna().all()
    assert merge_enrichment(out, None, CleaningReport())["title_synonyms"].map(len).sum() == 0


def test_english_titles_and_synonyms_resolve() -> None:
    index = TitleIndex(
        [
            TitleEntry(16498, "Shingeki no Kyojin", 4_000_000, ("Attack on Titan", "AoT")),
            TitleEntry(1535, "Death Note", 4_100_000, ("DN",)),
        ]
    )
    assert index.resolve("attack on titan").anime_id == 16498  # type: ignore[union-attr]
    assert index.resolve("AoT").anime_id == 16498  # type: ignore[union-attr]
    assert [e.anime_id for e in index.suggest("titan")] == [16498]


def test_circuit_breaker_stops_when_jikan_is_down(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anime_rec.ingestion.jikan as jikan

    monkeypatch.setattr(jikan, "MAX_CONSECUTIVE_FAILURES", 2)
    client, calls = client_for({i: [httpx.Response(504)] * 3 for i in (2, 1535, 7)})
    with pytest.raises(JikanUnavailableError, match="looks down"):
        run_enrich(settings, client=client)
    assert sorted(set(calls)) == [2, 1535]  # stopped before the third anime
