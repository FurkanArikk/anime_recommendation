"""One chat turn: message (+ UI filters and picks) -> understood intent -> recommendations.

Routing:
  titles only, one liked   -> similar(anime)                       "anything like Monster?"
  titles + text / several  -> taste_profile(liked, disliked, text)  "like Death Note but funnier"
  text only                -> search(text)                          "cozy camping show"
If the LLM parser is unavailable the whole message is used as a semantic search, so the
chat keeps working (less cleverly) without Gemini.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from anime_rec.log import get_logger
from anime_rec.recommender.filters import SearchFilters
from anime_rec.recommender.intent import Intent, TitleMention, merge_filters, validated_filters
from anime_rec.recommender.schemas import AnimeRef, Recommendation

if TYPE_CHECKING:
    from anime_rec.recommender.service import RecommenderService

log = get_logger(__name__)

CANDIDATES_FOR_EXPLAIN = 2


class Understood(BaseModel):
    """What the assistant did with the message, shown to the user for transparency."""

    mode: Literal["search", "similar", "taste"]
    query: str | None = None
    liked: list[AnimeRef] = Field(default_factory=list)
    disliked: list[AnimeRef] = Field(default_factory=list)
    unresolved_titles: list[str] = Field(default_factory=list)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    parsed_by_llm: bool = False


def _resolve(
    service: "RecommenderService", mentions: list[TitleMention], unresolved: list[str]
) -> list[int]:
    ids: list[int] = []
    for mention in mentions:
        entry = None
        for candidate in (mention.mal_title, mention.as_written):
            if candidate and (entry := service.titles.resolve(candidate)):
                break
        if entry is None:
            unresolved.append(mention.as_written)
        else:
            ids.append(entry.anime_id)
    return ids


def respond(
    service: "RecommenderService",
    message: str,
    *,
    intent: Intent | None,
    filters: SearchFilters | None = None,
    liked_ids: list[int] | None = None,
    disliked_ids: list[int] | None = None,
    limit: int = 6,
    explain: bool = True,
) -> tuple[Recommendation, Understood]:
    unresolved: list[str] = []
    liked = list(liked_ids or [])
    disliked = list(disliked_ids or [])
    merged = filters or SearchFilters()
    query: str | None = message
    if intent is not None:
        liked += _resolve(service, intent.liked, unresolved)
        disliked += _resolve(service, intent.disliked, unresolved)
        merged = merge_filters(merged, validated_filters(intent.filters, service.facets))
        query = intent.query
    liked = list(dict.fromkeys(liked))
    disliked = [i for i in dict.fromkeys(disliked) if i not in liked]

    pool = limit * (CANDIDATES_FOR_EXPLAIN if explain else 1)
    mode: Literal["search", "similar", "taste"]
    if liked and len(liked) == 1 and not disliked and not query:
        mode, hits = "similar", service.similar(liked[0], merged, limit=pool)
    elif liked:
        mode = "taste"
        hits = service.taste_profile(liked, disliked, merged, limit=pool, modifier=query)
    else:
        mode, hits = "search", service.search(query or message, merged, limit=pool)

    seeds = service.get_many([*liked, *disliked]) if liked or disliked else []
    understood = Understood(
        mode=mode,
        query=query,
        liked=[AnimeRef.of(s) for s in seeds[: len(liked)]],
        disliked=[AnimeRef.of(s) for s in seeds[len(liked) :]],
        unresolved_titles=unresolved,
        filters=merged,
        parsed_by_llm=intent is not None,
    )
    log.info("chat turn", mode=mode, liked=len(liked), disliked=len(disliked),
             unresolved=unresolved, llm=intent is not None)  # fmt: skip
    rec = (
        service.explain(message, hits, top_n=limit)
        if explain
        else Recommendation(items=hits[:limit])
    )
    # Sequels/prequels of liked anime are excluded from the results; point to them instead.
    rec.franchises = service.franchise_notes(seeds[: len(liked)])
    return rec, understood
