"""User-facing search filters -> Qdrant payload filter.

Tags (`include_tags` / `exclude_tags`) match across genres, themes and demographics, so a
user can ask for "Psychological" (a MAL theme) or "Seinen" (a demographic) the same way as
"Drama" (a genre). Missing values never pass a numeric filter: an anime with unknown
episode count is excluded by `max_episodes`, not treated as 0.
"""

from pydantic import BaseModel, Field, model_validator
from qdrant_client import models

TAG_FIELDS = ("genres", "themes", "demographics")


class SearchFilters(BaseModel):
    min_score: float | None = Field(default=None, ge=0, le=10)
    year_min: int | None = Field(default=None, ge=1900, le=2100)
    year_max: int | None = Field(default=None, ge=1900, le=2100)
    types: list[str] = Field(default_factory=list, description="e.g. TV, Movie, OVA")
    max_episodes: int | None = Field(default=None, ge=1)
    include_tags: list[str] = Field(
        default_factory=list, description="all must match (genre, theme or demographic)"
    )
    exclude_tags: list[str] = Field(default_factory=list, description="none may match")
    exclude_ongoing: bool = False

    @model_validator(mode="after")
    def _year_range(self) -> "SearchFilters":
        if self.year_min and self.year_max and self.year_min > self.year_max:
            raise ValueError("year_min must be <= year_max")
        return self

    def is_empty(self) -> bool:
        return self == SearchFilters()


def _tag_matches(tag: str) -> models.Filter:
    return models.Filter(
        should=[
            models.FieldCondition(key=field, match=models.MatchValue(value=tag))
            for field in TAG_FIELDS
        ]
    )


def to_qdrant_filter(
    filters: SearchFilters | None, exclude_ids: list[int] | None = None
) -> models.Filter | None:
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []
    f = filters or SearchFilters()

    if f.min_score is not None:
        must.append(models.FieldCondition(key="score", range=models.Range(gte=f.min_score)))
    if f.year_min is not None or f.year_max is not None:
        must.append(
            models.FieldCondition(
                key="start_year", range=models.Range(gte=f.year_min, lte=f.year_max)
            )
        )
    if f.types:
        must.append(models.FieldCondition(key="type", match=models.MatchAny(any=f.types)))
    if f.max_episodes is not None:
        must.append(models.FieldCondition(key="episodes", range=models.Range(lte=f.max_episodes)))
    must.extend(_tag_matches(tag) for tag in f.include_tags)  # each tag required
    if f.exclude_tags:
        must_not.extend(
            models.FieldCondition(key=field, match=models.MatchAny(any=f.exclude_tags))
            for field in TAG_FIELDS
        )
    if f.exclude_ongoing:
        must.append(models.FieldCondition(key="is_ongoing", match=models.MatchValue(value=False)))
    if exclude_ids:
        must_not.append(models.HasIdCondition(has_id=list(exclude_ids)))

    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)
