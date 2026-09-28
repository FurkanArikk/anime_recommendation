"""Ingest stage: read raw CSVs -> validate -> clean -> validate -> write parquet.

Idempotent: the output depends only on the raw files, and it is written to a temp file and
atomically renamed, so a crash never leaves a half-written parquet behind.
"""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from anime_rec.config import Settings
from anime_rec.ingestion import schema
from anime_rec.ingestion.clean import CleaningReport, RawTables, clean_anime
from anime_rec.log import get_logger

log = get_logger(__name__)

# file name -> (schema, dtypes). Dtypes are explicit so pandas never guesses
# (e.g. a title like "1" must stay a string, missing episodes must stay NA, not float).
_ID = "int64"
_TABLES: dict[str, tuple[str, pa.DataFrameSchema, dict[str, str]]] = {
    "anime": (
        "anime.csv",
        schema.RAW_ANIME,
        {
            "anime_id": _ID, "title": "str", "score": "float64", "rank": _ID,
            "popularity": _ID, "members": _ID, "synopsis": "str", "start_date": "str",
            "end_date": "str", "type": "str", "episodes": "Int64", "image_url": "str",
        },
    ),
    "genres": ("anime_genres.csv", schema.RAW_GENRES, {"anime_id": _ID, "genre": "str"}),
    "companies": (
        "anime_companies.csv",
        schema.RAW_COMPANIES,
        {"anime_id": _ID, "company_id": _ID, "role": "str"},
    ),
    "entities": (
        "entities.csv",
        schema.RAW_ENTITIES,
        {"entity_id": _ID, "entity_type": "str", "name": "str", "image_url": "str"},
    ),
    "characters": (
        "anime_characters.csv",
        schema.RAW_CHARACTERS,
        {"anime_id": _ID, "character_id": _ID, "role": "str"},
    ),
    "voice_actors": (
        "anime_voice_actors.csv",
        schema.RAW_VOICE_ACTORS,
        {"character_id": _ID, "person_id": _ID, "language": "str"},
    ),
    "staff": (
        "anime_staff.csv",
        schema.RAW_STAFF,
        {"anime_id": _ID, "person_id": _ID, "role": "str"},
    ),
}  # fmt: skip


def load_raw(raw_dir: Path) -> RawTables:
    frames: dict[str, pd.DataFrame] = {}
    for key, (file_name, table_schema, dtypes) in _TABLES.items():
        path = raw_dir / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"missing raw table {path}; download the dataset into {raw_dir}"
            )
        df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""])
        frames[key] = table_schema.validate(df, lazy=True)
        log.info("loaded raw table", table=key, rows=len(df))
    return RawTables(**frames)


def run_ingest(settings: Settings) -> Path:
    raw = load_raw(settings.raw_dir)
    report = CleaningReport()
    clean = schema.CLEAN_ANIME.validate(clean_anime(raw, report), lazy=True)
    log.info("cleaning report", **report.counts)

    out = settings.processed_parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    clean.to_parquet(tmp, index=False)
    tmp.replace(out)
    log.info("wrote processed parquet", path=str(out), rows=len(clean), columns=len(clean.columns))
    return out


def load_processed(settings: Settings) -> pd.DataFrame:
    """Read the cleaned parquet, restoring Python lists (pyarrow returns numpy arrays)."""
    path = settings.processed_parquet
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `anime-rec ingest` first")
    df = pd.read_parquet(path)
    for col in schema.LIST_COLUMNS:
        df[col] = df[col].map(list)
    return df
