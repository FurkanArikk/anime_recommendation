import pandas as pd
import pytest

from anime_rec.embeddings.documents import build_documents, full_document, synopsis_only_document

DEATH_NOTE = {
    "anime_id": 1535,
    "title": "Death Note",
    "type": "TV",
    "episodes": 37,
    "start_year": 2006,
    "end_year": 2007,
    "is_ongoing": False,
    "genres": ["Supernatural", "Suspense"],
    "themes": ["Psychological"],
    "demographics": ["Shounen"],
    "studios": ["Madhouse"],
    "synopsis": "A student finds a notebook that kills.",
    "score": 8.62,
    "members": 4_000_000,
}


def test_full_document_layout() -> None:
    assert full_document(DEATH_NOTE) == (
        "Title: Death Note\n"
        "Format: TV, 37 episodes (2006-2007)\n"
        "Genres: Supernatural, Suspense\n"
        "Themes: Psychological\n"
        "Demographic: Shounen\n"
        "Studio: Madhouse\n"
        "Synopsis: A student finds a notebook that kills."
    )


def test_volatile_fields_not_embedded() -> None:
    # Score/members change on every scrape; embedding them would invalidate the cache.
    doc = full_document(DEATH_NOTE)
    assert "8.62" not in doc and "4000000" not in doc


def test_empty_fields_omitted_and_missing_values_tolerated() -> None:
    sparse = {
        **DEATH_NOTE,
        "type": "Movie",
        "episodes": pd.NA,
        "start_year": pd.NA,
        "end_year": pd.NA,
        "genres": [],
        "themes": [],
        "demographics": [],
        "studios": [],
        "synopsis": None,
    }
    assert full_document(sparse) == "Title: Death Note\nFormat: Movie"


@pytest.mark.parametrize(
    ("overrides", "expected_format"),
    [
        ({"is_ongoing": True, "end_year": pd.NA, "episodes": pd.NA}, "Format: TV (2006-ongoing)"),
        ({"type": "Movie", "episodes": 1, "end_year": 2006}, "Format: Movie (2006)"),
    ],
)
def test_format_line(overrides: dict[str, object], expected_format: str) -> None:
    assert expected_format in full_document({**DEATH_NOTE, **overrides}).splitlines()


def test_synopsis_only_falls_back_to_title() -> None:
    assert synopsis_only_document(DEATH_NOTE) == "A student finds a notebook that kills."
    assert synopsis_only_document({**DEATH_NOTE, "synopsis": None}) == "Death Note"


def test_documents_are_deterministic_and_ordered() -> None:
    df = pd.DataFrame([DEATH_NOTE, {**DEATH_NOTE, "anime_id": 2, "title": "Other"}])
    docs = build_documents(df, "full")
    assert docs == build_documents(df, "full")
    assert docs[1].startswith("Title: Other")


def test_unknown_template() -> None:
    with pytest.raises(ValueError, match="unknown template"):
        build_documents(pd.DataFrame([DEATH_NOTE]), "nope")
