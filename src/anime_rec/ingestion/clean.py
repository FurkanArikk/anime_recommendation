"""Pure cleaning functions: raw MAL tables in, one denormalized row per anime out.

No I/O happens here, so every step is unit-testable with tiny in-memory frames.
See `pipeline.py` for loading, validation and writing.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

MAX_CHARACTERS = 6
CHARACTER_ROLES = ("Main", "Supporting")  # other values are scraper noise ("Unknown", staff roles)
# The scraper stored MAL's "None found, add some" UI text as two fake companies.
PLACEHOLDER_COMPANY_NAMES = frozenset({"None found", "add some"})
GENRE_KINDS = {"Genre": "genres", "Theme": "themes", "Demographic": "demographics"}

_SYNOPSIS_PLACEHOLDER = re.compile(r"^No synopsis( information)? has been added", re.IGNORECASE)
_SYNOPSIS_BOILERPLATE = re.compile(r"\[Written by MAL Rewrite\]|\((?:Source|source):[^)]*\)")
# Stripped hyperlinks leave a space before punctuation: "Sequel to Chainsaw Man ."
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)])")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class RawTables:
    anime: pd.DataFrame
    genres: pd.DataFrame
    companies: pd.DataFrame
    entities: pd.DataFrame
    characters: pd.DataFrame
    voice_actors: pd.DataFrame
    staff: pd.DataFrame


@dataclass
class CleaningReport:
    """Counts of what each step changed; logged so the pipeline's effect is auditable."""

    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, n: int) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(n)


def clean_synopsis(text: object) -> str | None:
    """Strip MAL boilerplate; return None for missing or placeholder synopses."""
    if not isinstance(text, str) or _SYNOPSIS_PLACEHOLDER.match(text.strip()):
        return None
    text = _SYNOPSIS_BOILERPLATE.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None


def parse_genre_label(label: str) -> tuple[str, str]:
    """Split kind prefix and undo the doubled name.

    'Theme::Gag Humor Gag Humor' -> ('themes', 'Gag Humor'); 'Action Action' -> ('genres', 'Action')
    """
    kind, sep, name = label.partition("::")
    if not sep:
        kind, name = "Genre", label
    if kind not in GENRE_KINDS:
        raise ValueError(f"unknown genre kind {kind!r} in label {label!r}")
    # The scraper concatenated each name with itself; undo it only when the halves match exactly.
    words = name.split(" ")
    half = len(words) // 2
    if len(words) % 2 == 0 and words[:half] == words[half:]:
        name = " ".join(words[:half])
    return GENRE_KINDS[kind], name.strip()


def year_from_iso(date: pd.Series) -> pd.Series:
    """All source dates are 'YYYY-01-01', i.e. only the year carries information."""
    return pd.to_numeric(date.str.slice(0, 4), errors="coerce").astype("Int64")


def _ordered_unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def build_genre_lists(genres: pd.DataFrame) -> pd.DataFrame:
    """One row per anime with sorted `genres`, `themes`, `demographics` lists."""
    parsed = genres["genre"].map(parse_genre_label)
    df = pd.DataFrame(
        {
            "anime_id": genres["anime_id"],
            "kind": parsed.str[0],
            "name": parsed.str[1],
        }
    ).drop_duplicates()
    out = (
        df.groupby(["anime_id", "kind"])["name"]
        .agg(list)
        .map(lambda names: sorted(set(names)))
        .unstack("kind")
        .reindex(columns=list(GENRE_KINDS.values()))
    )
    return out


def build_company_lists(companies: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    """`studios`, `producers`, `licensors` per anime, source order kept (primary studio first)."""
    names = entities.set_index("entity_id")["name"]
    df = companies.assign(name=companies["company_id"].map(names))
    df = df[df["name"].notna() & ~df["name"].isin(PLACEHOLDER_COMPANY_NAMES)]
    out = (
        df.groupby(["anime_id", "role"], sort=False)["name"]
        .agg(list)
        .map(_ordered_unique)
        .unstack("role")
        .reindex(columns=["Studio", "Producer", "Licensor"])
    )
    out.columns = ["studios", "producers", "licensors"]
    return out


def build_staff_lists(staff: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    """`directors` and `original_creators`. Roles are comma-joined, so match whole tokens:
    'Episode Director' and 'Sound Director' must not count as 'Director'."""
    names = entities.set_index("entity_id")["name"]
    df = staff.assign(
        name=staff["person_id"].map(names),
        role=staff["role"].str.split(","),
    ).explode("role")
    df["role"] = df["role"].str.strip()
    df = df[df["name"].notna()]
    wanted = {"Director": "directors", "Original Creator": "original_creators"}
    df = df[df["role"].isin(wanted)]
    out = (
        df.groupby(["anime_id", "role"], sort=False)["name"]
        .agg(list)
        .map(_ordered_unique)
        .unstack("role")
        .reindex(columns=list(wanted))
    )
    out.columns = list(wanted.values())
    return out


def build_main_characters(
    characters: pd.DataFrame,
    entities: pd.DataFrame,
    voice_actors: pd.DataFrame,
    limit: int = MAX_CHARACTERS,
) -> pd.Series:
    """Up to `limit` characters per anime (Main before Supporting), each as
    {name, role, image_url, voice_actor}. Nameless placeholders and duplicates are dropped."""
    ent = entities.set_index("entity_id")
    df = characters[characters["role"].isin(CHARACTER_ROLES)].drop_duplicates(
        ["anime_id", "character_id"]
    )
    df = df.assign(
        name=df["character_id"].map(ent["name"]),
        image_url=df["character_id"].map(ent["image_url"]),
    )
    df = df[df["name"].notna() & (df["name"].str.strip() != "")]

    first_va = voice_actors.drop_duplicates("character_id").set_index("character_id")["person_id"]
    df["voice_actor"] = df["character_id"].map(first_va).map(ent["name"])

    # Stable sort keeps the scraper's order within each role.
    df = df.assign(_main=df["role"] != "Main").sort_values(["anime_id", "_main"], kind="stable")
    df = df.groupby("anime_id", sort=False).head(limit)

    def to_record(row: pd.Series) -> dict[str, str | None]:
        return {
            "name": row["name"],
            "role": row["role"],
            "image_url": row["image_url"] if pd.notna(row["image_url"]) else None,
            "voice_actor": row["voice_actor"] if pd.notna(row["voice_actor"]) else None,
        }

    if df.empty:
        return pd.Series(dtype=object, name="main_characters")
    records = df.apply(to_record, axis=1)
    return records.groupby(df["anime_id"], sort=False).agg(list).rename("main_characters")


def clean_anime(raw: RawTables, report: CleaningReport | None = None) -> pd.DataFrame:
    """Join and clean all raw tables into one row per anime (column order = CLEAN_ANIME)."""
    report = report if report is not None else CleaningReport()

    anime = raw.anime.drop_duplicates()
    report.add("duplicate_anime_rows_dropped", len(raw.anime) - len(anime))
    dup_ids = anime["anime_id"][anime["anime_id"].duplicated()]
    if not dup_ids.empty:
        # Same id, different content: no safe automatic choice, so stop.
        raise ValueError(f"conflicting rows for anime_id(s): {sorted(dup_ids.unique())[:10]}")

    synopsis = anime["synopsis"].map(clean_synopsis)
    report.add("synopsis_missing_or_placeholder", synopsis.isna().sum())
    report.add(
        "synopsis_modified",
        (synopsis.notna() & (synopsis != anime["synopsis"])).sum(),
    )

    out = pd.DataFrame(
        {
            "anime_id": anime["anime_id"].astype(int),
            "title": anime["title"].str.strip(),
            "synopsis": synopsis.astype(object),
            "has_synopsis": synopsis.notna(),
            "type": anime["type"],
            "episodes": anime["episodes"].astype("Int64"),
            "start_year": year_from_iso(anime["start_date"]),
            "end_year": year_from_iso(anime["end_date"]),
            "is_ongoing": anime["end_date"].isna() | anime["episodes"].isna(),
            "score": anime["score"].astype(float),
            "rank": anime["rank"].astype(int),
            "popularity": anime["popularity"].astype(int),
            "members": anime["members"].astype(int),
            "image_url": anime["image_url"],
            "mal_url": "https://myanimelist.net/anime/" + anime["anime_id"].astype(str),
        }
    ).set_index("anime_id", drop=False)

    lists = pd.concat(
        [
            build_genre_lists(raw.genres),
            build_company_lists(raw.companies, raw.entities),
            build_staff_lists(raw.staff, raw.entities),
            build_main_characters(raw.characters, raw.entities, raw.voice_actors),
        ],
        axis=1,
    )
    out = out.join(lists, how="left").reset_index(drop=True)
    for col in lists.columns:
        out[col] = out[col].map(lambda v: v if isinstance(v, list) else [])

    no_genre = out["genres"].map(len) == 0
    no_tags = no_genre & (out["themes"].map(len) == 0) & (out["demographics"].map(len) == 0)
    report.add("anime_without_main_genre", no_genre.sum())
    report.add("anime_without_any_genre_tag", no_tags.sum())
    report.add("anime_without_studio", (out["studios"].map(len) == 0).sum())
    report.add("anime_without_characters", (out["main_characters"].map(len) == 0).sum())
    report.add("rows_out", len(out))

    return out.sort_values(["rank", "anime_id"], kind="stable").reset_index(drop=True)
