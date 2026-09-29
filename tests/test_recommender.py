"""Recommender tests: in-memory Qdrant + a deterministic keyword embedder (no model, no API)."""

from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from anime_rec.embeddings.base import Embedder, TaskType
from anime_rec.recommender.filters import SearchFilters, to_qdrant_filter
from anime_rec.recommender.schemas import AnimeHit
from anime_rec.recommender.service import (
    AnimeNotFoundError,
    RecommenderService,
    hybrid_rerank,
    quality_prior,
    tag_overlap,
)
from anime_rec.recommender.titles import TitleEntry, TitleIndex, franchise_key, same_franchise

NAME = "anime_test"
VOCAB = ("titan", "wall", "detective", "notebook", "camp", "girls", "boxing", "time")


class KeywordEmbedder(Embedder):
    """Vector = which vocabulary words occur: similarity is predictable in tests."""

    model = "keyword"
    dim = len(VOCAB)
    batch_size = 32

    def __init__(self) -> None:
        super().__init__(cache=None)
        self.queries: list[str] = []

    def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        if task_type is TaskType.QUERY:
            self.queries.extend(texts)
        out = []
        for text in texts:
            v = np.array([float(w in text.lower()) for w in VOCAB]) + 0.01
            out.append((v / np.linalg.norm(v)).tolist())
        return out


def anime(
    anime_id: int,
    title: str,
    synopsis: str,
    *,
    genres: tuple[str, ...] = (),
    themes: tuple[str, ...] = (),
    type_: str = "TV",
    episodes: int | None = 12,
    year: int = 2015,
    score: float = 8.0,
    members: int = 1000,
) -> dict[str, Any]:
    return {
        "anime_id": anime_id, "title": title, "synopsis": synopsis, "type": type_,
        "episodes": episodes, "start_year": year, "end_year": year, "is_ongoing": False,
        "score": score, "rank": anime_id, "popularity": anime_id, "members": members,
        "image_url": f"https://cdn.myanimelist.net/images/anime/{anime_id}.jpg",
        "mal_url": f"https://myanimelist.net/anime/{anime_id}",
        "genres": list(genres), "themes": list(themes), "demographics": ["Shounen"],
        "studios": ["Studio"], "directors": [], "original_creators": [], "main_characters": [],
    }  # fmt: skip


CATALOG = [
    anime(1, "Shingeki no Kyojin", "giants titan attack the wall", genres=("Action",),
          themes=("Gore",), episodes=25, year=2013, score=8.5, members=4_000_000),
    anime(2, "Shingeki no Kyojin Season 2", "titan wall mystery deepens", genres=("Action",),
          year=2017, score=8.5),
    anime(3, "Shingeki no Kyojin Season 3 Part 2", "titan wall basement", genres=("Action",),
          episodes=10, year=2019, score=9.0),
    anime(4, "Kabaneri", "zombie titan wall train", genres=("Action",), year=2016, score=7.3),
    anime(5, "Death Note", "notebook detective battle of wits", genres=("Suspense",),
          themes=("Psychological",), episodes=37, year=2006, score=8.6, members=4_100_000),
    anime(6, "Monster", "surgeon detective hunts killer", genres=("Suspense",),
          themes=("Psychological",), episodes=74, year=2004, score=8.9),
    anime(7, "Yuru Camp", "girls camp in winter", genres=("Slice of Life",), year=2018),
    anime(8, "Hajime no Ippo", "boxing underdog", genres=("Sports",), episodes=None, year=2000),
    anime(9, "Kimi no Na wa.", "time girls swap bodies", type_="Movie", episodes=1, year=2016),
]  # fmt: skip


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def service(embedder: KeywordEmbedder) -> RecommenderService:
    client = QdrantClient(":memory:")
    client.create_collection(
        NAME, vectors_config=models.VectorParams(size=len(VOCAB), distance=models.Distance.COSINE)
    )
    vectors = embedder.embed([a["synopsis"] for a in CATALOG], TaskType.DOCUMENT)
    client.upsert(
        NAME,
        points=[
            models.PointStruct(id=a["anime_id"], vector=v, payload=a)
            for a, v in zip(CATALOG, vectors, strict=True)
        ],
    )
    embedder.queries.clear()
    return RecommenderService(client, NAME, embedder)


def ids(hits: list[Any]) -> list[int]:
    return [h.anime_id for h in hits]


# --- titles -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "key"),
    [
        ("Shingeki no Kyojin Season 3 Part 2", "shingeki no kyojin"),
        ("Shingeki no Kyojin: The Final Season", "shingeki no kyojin"),
        ("Hajime no Ippo: New Challenger", "hajime no ippo"),
        ("Kingdom 6th Season", "kingdom"),
        ("Mob Psycho 100 II", franchise_key("Mob Psycho 100")),
        ("Gintama°", "gintama"),
        ("Yuru Camp△ Season 2", "yuru camp"),
        ("86", "86"),
        ("Yuru Camp△ Season 2 Specials", "yuru camp"),
        ("Yuru Camp△ Specials", "yuru camp"),
        ("Yama no Susume Third Season", "yama no susume"),
    ],
)
def test_franchise_key(title: str, key: str) -> None:
    assert franchise_key(title) == key


def test_title_resolution() -> None:
    index = TitleIndex(
        [
            TitleEntry(1, "Shingeki no Kyojin", 4_000_000),
            TitleEntry(2, "Shingeki no Kyojin Season 2", 2_000_000),
            TitleEntry(5, "Death Note", 4_100_000),
            TitleEntry(7, "Yuru Camp△", 900_000),
        ]
    )
    assert index.resolve("death note").anime_id == 5  # type: ignore[union-attr]
    assert index.resolve("Yuru Camp").anime_id == 7  # type: ignore[union-attr]  # symbols
    assert index.resolve("shingeki").anime_id == 1  # type: ignore[union-attr]  # popular prefix
    assert index.resolve("deth note").anime_id == 5  # type: ignore[union-attr]  # typo
    assert index.resolve("Attack on Titan") is None  # no English titles in the data
    assert [e.anime_id for e in index.suggest("kyojin")] == [1, 2]


# --- filters ------------------------------------------------------------------


def test_empty_filters_produce_no_filter() -> None:
    assert to_qdrant_filter(SearchFilters()) is None
    assert to_qdrant_filter(None) is None


def test_invalid_year_range_rejected() -> None:
    with pytest.raises(ValueError, match="year_min"):
        SearchFilters(year_min=2020, year_max=2010)


def test_filters_applied_inside_search(service: RecommenderService) -> None:
    hits = service.search("detective", SearchFilters(min_score=8.7))
    assert ids(hits)[0] == 6  # Monster (8.9); Death Note (8.6) filtered out
    assert 5 not in ids(hits) and all(h.score >= 8.7 for h in hits)
    hits = service.search("girls", SearchFilters(types=["Movie"]))
    assert ids(hits) == [9]


def test_tags_match_themes_and_genres(service: RecommenderService) -> None:
    # "Psychological" is a theme, not a genre, and still filters.
    assert set(ids(service.search("detective", SearchFilters(include_tags=["Psychological"])))) == {
        5,
        6,
    }
    hits = service.search("detective", SearchFilters(exclude_tags=["Psychological"]), limit=20)
    assert not {5, 6} & set(ids(hits))


def test_unknown_episode_count_does_not_pass_max_episodes(service: RecommenderService) -> None:
    hits = service.search("boxing", SearchFilters(max_episodes=100), limit=20)
    assert 8 not in ids(hits)  # Hajime no Ippo has episodes = None


def test_year_range(service: RecommenderService) -> None:
    hits = service.search("titan wall", SearchFilters(year_min=2016, year_max=2018), limit=20)
    assert set(ids(hits)) <= {2, 4, 7, 9}


# --- retrieval ----------------------------------------------------------------


def test_search_uses_query_task_and_collapses_franchises(
    service: RecommenderService, embedder: KeywordEmbedder
) -> None:
    hits = service.search("titan wall", limit=3)
    assert embedder.queries == ["titan wall"]
    titles = [h.title for h in hits]
    assert sum(t.startswith("Shingeki") for t in titles) == 1  # one entry per franchise
    assert "Kabaneri" in titles
    assert all(h.similarity is not None for h in hits)


def test_similar_excludes_seed_and_its_franchise(service: RecommenderService) -> None:
    hits = service.similar(1, limit=3)
    assert ids(hits)[0] == 4  # Kabaneri, not another Shingeki season
    assert not {1, 2, 3} & set(ids(hits))


def test_taste_profile_excludes_inputs(service: RecommenderService) -> None:
    hits = service.taste_profile(liked=[5], disliked=[7], limit=3)
    assert ids(hits)[0] == 6  # Monster: detective + psychological
    assert not {5, 7} & set(ids(hits))


def test_taste_profile_requires_liked(service: RecommenderService) -> None:
    with pytest.raises(ValueError, match="liked"):
        service.taste_profile(liked=[], disliked=[5])


def test_unknown_ids_raise(service: RecommenderService) -> None:
    with pytest.raises(AnimeNotFoundError):
        service.similar(12345)
    with pytest.raises(AnimeNotFoundError):
        service.taste_profile(liked=[5, 999])


def test_resolve_title_via_qdrant_backed_index(service: RecommenderService) -> None:
    assert service.resolve_title("death note").anime_id == 5
    assert len(service.titles) == len(CATALOG)


def test_explain_without_explainer_returns_retrieval_order(service: RecommenderService) -> None:
    hits = service.search("titan wall", limit=3)
    rec = service.explain("titan stuff", hits, top_n=2)
    assert not rec.explained and ids(rec.items) == ids(hits)[:2]


def test_hybrid_rerank_prefers_shared_tags_and_quality() -> None:
    seed = AnimeHit.from_payload(CATALOG[4])  # Death Note: Suspense + Psychological
    plot_twin = AnimeHit.from_payload({**CATALOG[6], "score": 6.0, "members": 500}, 0.80)
    tone_match = AnimeHit.from_payload(CATALOG[5], 0.72)  # Monster: same tags, 8.9
    assert tag_overlap(tone_match, [seed]) > tag_overlap(plot_twin, [seed])
    assert quality_prior(tone_match) > quality_prior(plot_twin)
    assert [h.anime_id for h in hybrid_rerank([plot_twin, tone_match], [seed])] == [6, 7]
    # Weights of zero reduce to plain cosine order.
    ordered = hybrid_rerank([tone_match, plot_twin], [seed], tag_weight=0, quality_weight=0)
    assert [h.anime_id for h in ordered] == [7, 6]


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("non non biyori", "non non biyori repeat", True),
        ("hajime no ippo", "hajime no ippo", True),
        ("monster", "monster musume", False),  # one-word keys must match exactly
        ("death note", "death march", False),
    ],
)
def test_same_franchise(a: str, b: str, same: bool) -> None:
    assert same_franchise(a, b) is same
