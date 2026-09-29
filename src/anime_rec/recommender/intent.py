"""Chat intent parsing: free text -> {semantic query, liked/disliked titles, filters}.

"something like Death Note but funnier, under 25 episodes" becomes
    query="funnier, comedic", liked=[Death Note], filters={include_tags:[Comedy], max_episodes:25}

The LLM only proposes; code validates. Tags and types are checked against the values that
actually exist in the catalog (so it cannot invent a filter), numbers are clamped, and
titles are resolved against our own title index, never trusted as-is.
"""

from pydantic import BaseModel, Field

from anime_rec.recommender.filters import SearchFilters
from anime_rec.recommender.llm import GeminiJson
from anime_rec.recommender.schemas import Facets

SYSTEM_INSTRUCTION = """\
You turn an anime recommendation request into a structured search.
Fields:
- query: what the user wants the anime to be ABOUT or FEEL like (plot, mood, setting,
  themes), rewritten as a short description. Do not include titles, numbers or filters.
  Use null if the message only names titles and/or filters.
- liked: anime the user names as examples they like or want something similar to.
- disliked: anime the user names as things to avoid.
  For each title give `as_written` exactly as the user wrote it and `mal_title`: the
  official MyAnimeList title (usually romaji, e.g. "Shingeki no Kyojin" for
  "Attack on Titan") if you know it, else null.
- filters: set ONLY what the user explicitly asks for.
  include_tags / exclude_tags must be chosen from ALLOWED TAGS (e.g. "funny" -> Comedy,
  "no romance" -> exclude Romance). An anime must have ALL include_tags, so add exactly one
  tag per genre the user names ("slice of life" -> Slice of Life) and never two tags for
  the same idea (not both Slice of Life and Iyashikei).
  types from ALLOWED TYPES. "short" means max_episodes 13; "recent" means year_min
  {recent_year}; "classic"/"old" means year_max 2005.
  Any quality word ("good", "great", "great story", "best", "highly rated", "masterpiece")
  means min_score 7.5.
"""


class TitleMention(BaseModel):
    as_written: str
    mal_title: str | None = None


class IntentFilters(BaseModel):
    min_score: float | None = None
    year_min: int | None = None
    year_max: int | None = None
    max_episodes: int | None = None
    types: list[str] = Field(default_factory=list)
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)


class Intent(BaseModel):
    query: str | None = None
    liked: list[TitleMention] = Field(default_factory=list)
    disliked: list[TitleMention] = Field(default_factory=list)
    filters: IntentFilters = Field(default_factory=IntentFilters)


def _clamp(value: float | None, lo: float, hi: float) -> float | None:
    return None if value is None else min(max(value, lo), hi)


def validated_filters(raw: IntentFilters, facets: Facets) -> SearchFilters:
    """Keep only values that exist in the catalog; fix ranges instead of failing."""
    tags = {
        f.value.casefold(): f.value for f in [*facets.genres, *facets.themes, *facets.demographics]
    }
    types = {f.value.casefold(): f.value for f in facets.type}

    def known(values: list[str], allowed: dict[str, str]) -> list[str]:
        return list(dict.fromkeys(allowed[v.casefold()] for v in values if v.casefold() in allowed))

    year_min, year_max = raw.year_min, raw.year_max
    if year_min and year_max and year_min > year_max:
        year_min, year_max = year_max, year_min
    episodes = raw.max_episodes if raw.max_episodes and raw.max_episodes >= 1 else None
    return SearchFilters(
        min_score=_clamp(raw.min_score, 0, 10),
        year_min=min(max(year_min, 1900), 2100) if year_min else None,
        year_max=min(max(year_max, 1900), 2100) if year_max else None,
        max_episodes=episodes,
        types=known(raw.types, types),
        include_tags=known(raw.include_tags, tags),
        exclude_tags=known(raw.exclude_tags, tags),
    )


def merge_filters(base: SearchFilters, extra: SearchFilters) -> SearchFilters:
    """Explicit UI filters win; the parsed intent fills what the UI left unset."""
    merged = base.model_copy()
    for field in ("min_score", "year_min", "year_max", "max_episodes"):
        if getattr(merged, field) is None:
            setattr(merged, field, getattr(extra, field))
    merged.types = base.types or extra.types
    merged.include_tags = list(dict.fromkeys([*base.include_tags, *extra.include_tags]))
    merged.exclude_tags = list(dict.fromkeys([*base.exclude_tags, *extra.exclude_tags]))
    if merged.year_min and merged.year_max and merged.year_min > merged.year_max:
        merged.year_min, merged.year_max = base.year_min, base.year_max
    return merged


class IntentParser:
    def __init__(self, llm: GeminiJson, recent_year: int = 2020) -> None:
        self._llm = llm
        self._recent_year = recent_year

    def parse(self, message: str, facets: Facets) -> Intent | None:
        tags = sorted(f.value for f in [*facets.genres, *facets.themes, *facets.demographics])
        types = [f.value for f in facets.type]
        prompt = (
            f"ALLOWED TAGS: {', '.join(tags)}\nALLOWED TYPES: {', '.join(types)}\n\n"
            f"USER MESSAGE:\n{message}"
        )
        return self._llm.generate(
            prompt,
            system=SYSTEM_INSTRUCTION.format(recent_year=self._recent_year),
            schema=Intent,
            temperature=0.0,
        )
