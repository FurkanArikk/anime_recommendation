"""Typed recommendation results (also the API response models)."""

from typing import Any

from pydantic import BaseModel, Field


class Character(BaseModel):
    name: str
    role: str
    image_url: str | None = None
    voice_actor: str | None = None


class AnimeHit(BaseModel):
    anime_id: int
    title: str
    title_english: str | None = None
    title_japanese: str | None = None
    title_synonyms: list[str] = Field(default_factory=list)
    synopsis: str | None = None
    type: str
    episodes: int | None = None
    start_year: int | None = None
    end_year: int | None = None
    is_ongoing: bool = False
    score: float
    rank: int
    popularity: int
    members: int
    image_url: str
    mal_url: str
    genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    demographics: list[str] = Field(default_factory=list)
    studios: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    original_creators: list[str] = Field(default_factory=list)
    main_characters: list[Character] = Field(default_factory=list)
    source: str | None = None
    age_rating: str | None = None
    season: str | None = None
    similarity: float | None = Field(default=None, description="cosine similarity to the query")
    reason: str | None = Field(default=None, description="LLM explanation, grounded in this entry")

    @classmethod
    def from_payload(cls, payload: dict[str, Any], similarity: float | None = None) -> "AnimeHit":
        return cls.model_validate({**payload, "similarity": similarity})


class AnimeRef(BaseModel):
    anime_id: int
    title: str
    title_english: str | None = None
    image_url: str | None = None
    members: int | None = None

    @classmethod
    def of(cls, hit: "AnimeHit") -> "AnimeRef":
        return cls(
            anime_id=hit.anime_id,
            title=hit.title,
            title_english=hit.title_english,
            image_url=hit.image_url,
            members=hit.members,
        )


class FranchiseEntry(BaseModel):
    anime_id: int
    title: str
    title_english: str | None = None
    image_url: str | None = None
    type: str
    start_year: int | None = None
    episodes: int | None = None
    score: float


class FranchiseNote(BaseModel):
    """Other entries (seasons, movies, OVAs) of an anime the user liked. Kept out of the
    recommendations on purpose (they'd crowd out discovery) but surfaced as a note."""

    seed: AnimeRef
    entries: list[FranchiseEntry]
    total: int = Field(description="entries in the franchise besides the seed, before limiting")


class Recommendation(BaseModel):
    items: list[AnimeHit]
    summary: str | None = None
    explained: bool = Field(default=False, description="True if the LLM re-ranked/explained")
    franchises: list[FranchiseNote] = Field(default_factory=list)


class FacetValue(BaseModel):
    value: str
    count: int


class Facets(BaseModel):
    genres: list[FacetValue] = Field(default_factory=list)
    themes: list[FacetValue] = Field(default_factory=list)
    demographics: list[FacetValue] = Field(default_factory=list)
    type: list[FacetValue] = Field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
