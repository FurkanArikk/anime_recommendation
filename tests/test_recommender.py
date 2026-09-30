"""Recommender tests: in-memory Qdrant + a deterministic keyword embedder (no model, no API)."""

from typing import Any

import pytest

from anime_rec.recommender.filters import SearchFilters, to_qdrant_filter
from anime_rec.recommender.schemas import AnimeHit
from anime_rec.recommender.service import (
    AnimeNotFoundError,
    RecommenderService,
    hybrid_rerank,
    quality_prior,
    tag_overlap,
)
from anime_rec.recommender.titles import (
    TitleEntry,
    TitleIndex,
    franchise_key,
    in_franchise,
    same_franchise,
)
from tests.fakes import CATALOG, KeywordEmbedder, make_service


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def service(embedder: KeywordEmbedder) -> RecommenderService:
    return make_service(embedder)


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


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        # Single-word names ending in a symbol are distinctive: all Haikyuu!! seasons match.
        ("Haikyuu!!", "Haikyuu!! Karasuno Koukou vs. Shiratorizawa Gakuen Koukou", True),
        ("Haikyuu!! To the Top Part 2", "Haikyuu!! Second Season", True),
        ("Gintama", "Gintama°", True),
        ("Death Note", "Death Note: Rewrite", True),
        ("Kingdom", "Kingdom 6th Season", True),
        (
            "Re:Zero kara Hajimeru Isekai Seikatsu",
            "Re:Zero kara Hajimeru Isekai Seikatsu 2nd Season",
            True,
        ),
        # Plain single words and shared prefixes must not merge unrelated shows.
        ("Monster", "Monster Musume no Iru Nichijou", False),
        ("Kingdom", "Kingdom Hearts χ Back Cover", False),
        ("Re:Zero kara Hajimeru Isekai Seikatsu", "Re:Creators", False),  # was a bug: key 're'
        ("Death Note", "Death Parade", False),
        ("Shingeki no Kyojin", "Shingeki no Bahamut: Genesis", False),
    ],
)
def test_in_franchise(a: str, b: str, same: bool) -> None:
    assert in_franchise(a, b) is same
    assert in_franchise(b, a) is same  # symmetric


def test_franchise_note_lists_other_entries_tv_first(service: RecommenderService) -> None:
    (seed,) = service.get_many([2])  # Shingeki no Kyojin Season 2
    note = service.franchise(seed)
    assert [e.anime_id for e in note.entries] == [1, 3]  # 2013, 2019: seed itself excluded
    assert note.total == 2 and note.seed.anime_id == 2


def test_franchise_notes_skip_standalone_anime(service: RecommenderService) -> None:
    seeds = service.get_many([5, 1])  # Death Note (standalone here), Shingeki no Kyojin
    notes = service.franchise_notes(seeds)
    assert [n.seed.anime_id for n in notes] == [1]
