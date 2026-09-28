"""Shared fixtures: a tiny, hand-built copy of the raw MAL tables that reproduces the
real data's quirks (doubled genre names, placeholder companies, duplicate rows, ...)."""

import pandas as pd
import pytest

from anime_rec.ingestion.clean import RawTables

LIGHT_IMG = "https://cdn.myanimelist.net/images/characters/light.jpg"
LONG_SYNOPSIS = "A long enough synopsis about a genius and a notebook. [Written by MAL Rewrite]"


@pytest.fixture
def raw_tables() -> RawTables:
    anime = pd.DataFrame(
        {
            "anime_id": [1535, 21, 99],
            "title": ["Death Note", "One Piece", "Mystery Movie"],
            "score": [8.62, 8.73, 6.5],
            "rank": [80, 50, 9000],
            "popularity": [2, 20, 8000],
            "members": [4_000_000, 2_500_000, 300],
            "synopsis": [
                LONG_SYNOPSIS,
                "Luffy sails the Grand Line , searching for treasure . (Source: ANN)",
                "No synopsis information has been added to this title. Help improve our database .",
            ],
            "start_date": ["2006-01-01", "1999-01-01", None],
            "end_date": ["2007-01-01", None, None],
            "type": ["TV", "TV", "Movie"],
            "episodes": pd.array([37, None, 1], dtype="Int64"),
            "image_url": [
                f"https://cdn.myanimelist.net/images/anime/{i}/{i}.jpg" for i in (1, 2, 3)
            ],
        }
    )
    # Exact duplicate row, as in the real data (anime 41884).
    anime = pd.concat([anime, anime.iloc[[0]]], ignore_index=True)

    genres = pd.DataFrame(
        {
            "anime_id": [1535, 1535, 1535, 1535, 1535, 21, 21],
            "genre": [
                "Supernatural Supernatural",
                "Suspense Suspense",
                "Suspense Suspense",  # duplicate row
                "Theme::Psychological Psychological",
                "Demographic::Shounen Shounen",
                "Action Action",
                "Slice of Life Slice of Life",
            ],
        }
    )
    entities = pd.DataFrame(
        {
            "entity_id": [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
            "entity_type": [
                "studio", "studio", "studio", "producer", "licensor",
                "character", "character", "voice_actor", "staff", "staff", "character",
            ],
            "name": [
                "Madhouse", "None found", "add some", "VAP", "VIZ Media",
                None, "Yagami, Light", "Miyano, Mamoru", "Araki, Tetsurou", "Ohba, Tsugumi", "L",
            ],
            "image_url": [
                None, None, None, None, None,
                None, LIGHT_IMG, None, None, None, None,
            ],
        }
    )  # fmt: skip
    companies = pd.DataFrame(
        {
            "anime_id": [1535, 1535, 1535, 99, 99],
            "company_id": [1, 4, 5, 2, 3],
            "role": ["Studio", "Producer", "Licensor", "Studio", "Studio"],
        }
    )
    characters = pd.DataFrame(
        {
            "anime_id": [1535, 1535, 1535, 1535, 1535],
            "character_id": [14, 9, 10, 9, 10],
            "role": ["Supporting", "Unknown", "Main", "Unknown", "Main"],
        }
    )
    voice_actors = pd.DataFrame({"character_id": [10], "person_id": [11], "language": ["Japanese"]})
    staff = pd.DataFrame(
        {
            "anime_id": [1535, 1535],
            "person_id": [12, 13],
            "role": ["Director, Storyboard, Episode Director", "Original Creator"],
        }
    )
    return RawTables(anime, genres, companies, entities, characters, voice_actors, staff)
