import pandas as pd
import pytest

from anime_rec.ingestion.clean import (
    RawTables,
    build_company_lists,
    build_staff_lists,
    clean_anime,
    clean_synopsis,
    parse_genre_label,
)
from anime_rec.ingestion.schema import CLEAN_ANIME


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Action Action", ("genres", "Action")),
        ("Slice of Life Slice of Life", ("genres", "Slice of Life")),
        ("Boys Love", ("genres", "Boys Love")),  # not doubled: left alone
        ("Theme::Gag Humor Gag Humor", ("themes", "Gag Humor")),
        ("Theme::Idols (Female) Idols (Female)", ("themes", "Idols (Female)")),
        ("Demographic::Shounen Shounen", ("demographics", "Shounen")),
    ],
)
def test_parse_genre_label(label: str, expected: tuple[str, str]) -> None:
    assert parse_genre_label(label) == expected


def test_parse_genre_label_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown genre kind"):
        parse_genre_label("Studio::Madhouse Madhouse")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sequel to Chainsaw Man .", "Sequel to Chainsaw Man."),
        ("Great story. [Written by MAL Rewrite]", "Great story."),
        ("Great story. (Source: GKIDS, edited)", "Great story."),
        ("Too   many\tspaces", "Too many spaces"),
        ("No synopsis information has been added to this title. Help improve .", None),
        ("No synopsis has been added for this series yet.", None),
        ("   ", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_clean_synopsis(raw: object, expected: str | None) -> None:
    assert clean_synopsis(raw) == expected


def test_company_placeholders_dropped_and_order_kept(raw_tables: RawTables) -> None:
    out = build_company_lists(raw_tables.companies, raw_tables.entities)
    assert out.loc[1535, "studios"] == ["Madhouse"]
    assert out.loc[1535, "licensors"] == ["VIZ Media"]
    # anime 99 only had the "None found" / "add some" placeholders
    assert 99 not in out.index


def test_director_matches_whole_role_token(raw_tables: RawTables) -> None:
    staff = pd.concat(
        [
            raw_tables.staff,
            pd.DataFrame({"anime_id": [1535], "person_id": [11], "role": ["Episode Director"]}),
        ]
    )
    out = build_staff_lists(staff, raw_tables.entities)
    assert out.loc[1535, "directors"] == ["Araki, Tetsurou"]
    assert out.loc[1535, "original_creators"] == ["Ohba, Tsugumi"]


@pytest.fixture
def clean(raw_tables: RawTables) -> pd.DataFrame:
    return clean_anime(raw_tables)


def test_output_satisfies_clean_schema(clean: pd.DataFrame) -> None:
    CLEAN_ANIME.validate(clean, lazy=True)


def test_duplicate_rows_dropped_and_sorted_by_rank(clean: pd.DataFrame) -> None:
    assert clean["anime_id"].tolist() == [21, 1535, 99]


def test_conflicting_duplicate_ids_raise(raw_tables: RawTables) -> None:
    conflicting = raw_tables.anime.iloc[[0]].assign(score=1.0)
    raw_tables.anime = pd.concat([raw_tables.anime, conflicting])
    with pytest.raises(ValueError, match="conflicting rows"):
        clean_anime(raw_tables)


def test_row_level_cleaning(clean: pd.DataFrame) -> None:
    death_note, one_piece, mystery = (clean.set_index("anime_id").loc[i] for i in (1535, 21, 99))

    assert death_note["genres"] == ["Supernatural", "Suspense"]
    assert death_note["themes"] == ["Psychological"]
    assert death_note["demographics"] == ["Shounen"]
    assert death_note["start_year"] == 2006 and death_note["end_year"] == 2007
    assert not death_note["is_ongoing"]
    assert death_note["mal_url"] == "https://myanimelist.net/anime/1535"
    assert death_note["synopsis"].endswith("notebook.")

    # Missing episodes stay NA (never 0, which would pass a "max episodes" filter).
    assert pd.isna(one_piece["episodes"]) and one_piece["is_ongoing"]
    assert one_piece["synopsis"] == "Luffy sails the Grand Line, searching for treasure."

    # Placeholder synopsis -> missing, but the anime is kept.
    assert pd.isna(mystery["synopsis"]) and not mystery["has_synopsis"]  # None or NaN (pandas 3)
    assert pd.isna(mystery["start_year"])
    assert mystery["genres"] == [] and mystery["studios"] == []


def test_main_characters(clean: pd.DataFrame) -> None:
    chars = clean.set_index("anime_id").loc[1535, "main_characters"]
    # Unknown placeholder (id 9) and duplicate Light dropped; Main before Supporting.
    assert [c["name"] for c in chars] == ["Yagami, Light", "L"]
    assert chars[0]["voice_actor"] == "Miyano, Mamoru"
    assert chars[0]["image_url"].endswith("light.jpg")
    assert chars[1]["image_url"] is None


def test_cleaning_is_deterministic(raw_tables: RawTables) -> None:
    pd.testing.assert_frame_equal(clean_anime(raw_tables), clean_anime(raw_tables))
