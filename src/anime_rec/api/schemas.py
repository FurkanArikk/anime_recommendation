"""HTTP request/response models. Result models are reused from the recommender."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from anime_rec.recommender.chat import Understood
from anime_rec.recommender.filters import SearchFilters
from anime_rec.recommender.schemas import AnimeHit, AnimeRef, FranchiseNote

MAX_LIMIT = 30


class _Request(BaseModel):
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=MAX_LIMIT)
    explain: bool = Field(
        default=False,
        description="Re-rank with Gemini and attach a grounded reason to each result (~3s)",
    )


class SearchRequest(_Request):
    query: str = Field(
        min_length=1,
        max_length=500,
        examples=["a dark psychological thriller with a smart protagonist"],
    )


class SimilarRequest(_Request):
    anime_id: int | None = Field(default=None, examples=[1535])
    title: str | None = Field(default=None, examples=["Death Note"])

    @model_validator(mode="after")
    def _one_seed(self) -> "SimilarRequest":
        if (self.anime_id is None) == (self.title is None):
            raise ValueError("provide exactly one of anime_id or title")
        return self


class RecommendRequest(_Request):
    liked: list[int] = Field(min_length=1, max_length=20, examples=[[9253, 19, 13601]])
    disliked: list[int] = Field(default_factory=list, max_length=20, examples=[[11757]])
    strategy: Literal["best_score", "average_vector", "sum_scores"] = "best_score"


class RecommendationResponse(BaseModel):
    items: list[AnimeHit]
    summary: str | None = None
    explained: bool = False
    seeds: list[AnimeRef] = Field(default_factory=list, description="resolved input anime")
    franchises: list[FranchiseNote] = Field(
        default_factory=list, description="other seasons/movies of liked anime (not in items)"
    )
    took_ms: int


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    collection: str
    points: int
    embedding_model: str
    chat_models: list[str]


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=1000,
        examples=["something like Death Note but funnier, under 25 episodes"],
    )
    filters: SearchFilters = Field(default_factory=SearchFilters, description="from UI controls")
    liked: list[int] = Field(default_factory=list, max_length=20)
    disliked: list[int] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=6, ge=1, le=MAX_LIMIT)
    explain: bool = True


class ChatResponse(BaseModel):
    items: list[AnimeHit]
    summary: str | None = None
    explained: bool = False
    understood: Understood
    franchises: list[FranchiseNote] = Field(default_factory=list)
    took_ms: int
