"""Pandera schemas for the raw MAL tables and the cleaned output.

Raw schemas describe what the scraper *should* produce, so a changed upstream format fails
loudly at load time instead of silently corrupting embeddings. The clean schema is the
contract every downstream stage (embeddings, indexing, API) relies on.
"""

import pandera.pandas as pa

ANIME_TYPES = ["TV", "Movie", "ONA", "OVA", "Special", "TV Special", "Music", "CM", "PV"]
COMPANY_ROLES = ["Studio", "Producer", "Licensor"]
ENTITY_TYPES = ["studio", "producer", "licensor", "character", "voice_actor", "staff"]

_id = pa.Column(int, pa.Check.gt(0))
_iso_date = pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"), nullable=True)
_is_list = pa.Check(lambda v: isinstance(v, list), element_wise=True, error="must be a list")

RAW_ANIME = pa.DataFrameSchema(
    {
        "anime_id": _id,
        "title": pa.Column(str, pa.Check.str_length(min_value=1)),
        "score": pa.Column(float, pa.Check.in_range(0, 10)),
        "rank": _id,
        "popularity": _id,
        "members": pa.Column(int, pa.Check.ge(0)),
        "synopsis": pa.Column(str, nullable=True),
        "start_date": _iso_date,
        "end_date": _iso_date,
        "type": pa.Column(str, pa.Check.isin(ANIME_TYPES)),
        "episodes": pa.Column("Int64", pa.Check.ge(1), nullable=True),
        "image_url": pa.Column(str, pa.Check.str_startswith("https://cdn.myanimelist.net/")),
    },
    strict=True,  # unexpected columns mean the scraper changed: fail rather than ignore
)

RAW_GENRES = pa.DataFrameSchema(
    {"anime_id": _id, "genre": pa.Column(str, pa.Check.str_length(min_value=1))}, strict=True
)

RAW_COMPANIES = pa.DataFrameSchema(
    {"anime_id": _id, "company_id": _id, "role": pa.Column(str, pa.Check.isin(COMPANY_ROLES))},
    strict=True,
)

RAW_ENTITIES = pa.DataFrameSchema(
    {
        "entity_id": pa.Column(int, pa.Check.gt(0), unique=True),
        "entity_type": pa.Column(str, pa.Check.isin(ENTITY_TYPES)),
        "name": pa.Column(str, nullable=True),
        "image_url": pa.Column(str, nullable=True),
    },
    strict=True,
)

RAW_CHARACTERS = pa.DataFrameSchema(
    {"anime_id": _id, "character_id": _id, "role": pa.Column(str)}, strict=True
)

RAW_VOICE_ACTORS = pa.DataFrameSchema(
    {"character_id": _id, "person_id": _id, "language": pa.Column(str)}, strict=True
)

RAW_STAFF = pa.DataFrameSchema(
    {"anime_id": _id, "person_id": _id, "role": pa.Column(str)}, strict=True
)

LIST_COLUMNS = [
    "genres",
    "themes",
    "demographics",
    "studios",
    "producers",
    "licensors",
    "directors",
    "original_creators",
    "main_characters",
]
# Every list-valued column in the cleaned output (parquet round-trips them as arrays).
ALL_LIST_COLUMNS = [*LIST_COLUMNS, "title_synonyms"]

CLEAN_ANIME = pa.DataFrameSchema(
    {
        "anime_id": pa.Column(int, pa.Check.gt(0), unique=True),
        "title": pa.Column(str, pa.Check.str_length(min_value=1)),
        "synopsis": pa.Column(str, pa.Check.str_length(min_value=1), nullable=True),
        "has_synopsis": pa.Column(bool),
        "type": pa.Column(str, pa.Check.isin(ANIME_TYPES)),
        "episodes": pa.Column("Int64", pa.Check.ge(1), nullable=True),
        "start_year": pa.Column("Int64", pa.Check.in_range(1900, 2100), nullable=True),
        "end_year": pa.Column("Int64", pa.Check.in_range(1900, 2100), nullable=True),
        "is_ongoing": pa.Column(bool),
        "score": pa.Column(float, pa.Check.in_range(0, 10)),
        "rank": pa.Column(int, pa.Check.gt(0)),
        "popularity": pa.Column(int, pa.Check.gt(0)),
        "members": pa.Column(int, pa.Check.ge(0)),
        "image_url": pa.Column(str),
        "mal_url": pa.Column(str, pa.Check.str_startswith("https://myanimelist.net/anime/")),
        **{col: pa.Column(object, _is_list) for col in LIST_COLUMNS},
        # Jikan enrichment (empty when the enrich stage hasn't run)
        "title_english": pa.Column(str, nullable=True),
        "title_japanese": pa.Column(str, nullable=True),
        "source": pa.Column(str, nullable=True),
        "age_rating": pa.Column(str, nullable=True),
        "season": pa.Column(str, nullable=True),
        "title_synonyms": pa.Column(object, _is_list),
    },
    checks=pa.Check(
        lambda df: (
            df["end_year"].isna() | df["start_year"].isna() | (df["end_year"] >= df["start_year"])
        ),
        error="end_year must not precede start_year",
    ),
    strict=True,
    ordered=True,
)
