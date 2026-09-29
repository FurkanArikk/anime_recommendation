"""Recommendation service: pure retrieval logic, no HTTP or UI concerns.

Four ways in, one result type:
  similar(anime_id)             nearest neighbours of one anime (its stored vector)
  search(text)                  natural-language request embedded as a RETRIEVAL_QUERY
  taste_profile(liked, disliked) Qdrant recommend API with positive/negative examples
  + SearchFilters on all three  payload filtering inside the vector search

Results are over-fetched and collapsed so one franchise (five seasons of the same show)
cannot fill the page, and anime the user supplied as input are never recommended back.

Item-based modes (similar, taste_profile) re-rank a larger candidate pool with a hybrid
score: cosine + tag overlap with the seeds + a small quality prior. Vectors are built from
synopses only (best for text queries in the evaluation), which captures plot premise but
not tone: without tags, "similar to Death Note" returned other *death god* shows rather
than psychological thrillers. Text search stays pure cosine, as measured.
"""

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from qdrant_client import QdrantClient, models

from anime_rec.embeddings.base import Embedder
from anime_rec.log import get_logger
from anime_rec.recommender import chat as chat_turn
from anime_rec.recommender.explain import Explainer
from anime_rec.recommender.filters import SearchFilters, to_qdrant_filter
from anime_rec.recommender.intent import IntentParser
from anime_rec.recommender.schemas import AnimeHit, Facets, FacetValue, Recommendation
from anime_rec.recommender.titles import TitleEntry, TitleIndex, franchise_key, same_franchise

log = get_logger(__name__)

OVERFETCH = 4  # candidates per requested result, to survive franchise collapsing
FACET_FIELDS = ("genres", "themes", "demographics", "type")
RERANK_POOL = 80  # candidates re-ranked in item-based modes
TAG_WEIGHT = 0.3
QUALITY_WEIGHT = 0.05
Strategy = Literal["average_vector", "best_score", "sum_scores"]


class AnimeNotFoundError(LookupError):
    pass


def _tags(hit: AnimeHit) -> set[str]:
    return {*hit.genres, *hit.themes, *hit.demographics}


def tag_overlap(hit: AnimeHit, seeds: Sequence[AnimeHit]) -> float:
    """Best Jaccard overlap of genre/theme/demographic tags with any seed (0..1)."""
    tags = _tags(hit)
    best = 0.0
    for seed in seeds:
        union = tags | _tags(seed)
        if union:
            best = max(best, len(tags & _tags(seed)) / len(union))
    return best


def quality_prior(hit: AnimeHit) -> float:
    """Roughly -1..+1: MAL score around 7 and audience size around 100k are neutral."""
    return 0.5 * (hit.score - 7.0) + 0.5 * (math.log10(max(hit.members, 1)) - 5.0)


def hybrid_rerank(
    hits: Sequence[AnimeHit],
    seeds: Sequence[AnimeHit],
    tag_weight: float = TAG_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
) -> list[AnimeHit]:
    def score(h: AnimeHit) -> float:
        tags = tag_weight * tag_overlap(h, seeds)
        return (h.similarity or 0.0) + tags + quality_weight * quality_prior(h)

    return sorted(hits, key=score, reverse=True)


def collapse_franchises(
    hits: Sequence[AnimeHit], limit: int, exclude_keys: set[str] | None = None
) -> list[AnimeHit]:
    """Keep the best-scoring entry per franchise (hits arrive best-first)."""
    seen = list(exclude_keys or ())
    out: list[AnimeHit] = []
    for hit in hits:
        key = franchise_key(hit.title)
        if any(same_franchise(key, k) for k in seen):
            continue
        seen.append(key)
        out.append(hit)
        if len(out) == limit:
            break
    return out


def _title_entry(anime_id: int, payload: dict[str, Any]) -> TitleEntry:
    aliases = [payload.get("title_english"), *(payload.get("title_synonyms") or [])]
    return TitleEntry(
        anime_id,
        payload["title"],
        int(payload["members"]),
        tuple(a for a in aliases if a and a != payload["title"]),
        english=payload.get("title_english"),
        image_url=payload.get("image_url"),
    )


class RecommenderService:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: Embedder,
        explainer: Explainer | None = None,
        intent_parser: IntentParser | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._explainer = explainer
        self._intent_parser = intent_parser
        self._titles: TitleIndex | None = None
        self._facets: Facets | None = None

    # --- introspection ----------------------------------------------------------

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def embedding_model(self) -> str:
        return self._embedder.model

    @property
    def chat_models(self) -> list[str]:
        return list(self._explainer.models) if self._explainer else []

    def count(self) -> int:
        return int(self._client.count(self._collection, exact=False).count)

    # --- lookup -----------------------------------------------------------------

    def _load_catalog(self) -> None:
        """One scroll over Qdrant (the serving source of truth; no parquet needed) builds
        the title index and the filter facets."""
        fields = ["title", "members", "title_english", "title_synonyms", "image_url", *FACET_FIELDS,
                  "start_year"]  # fmt: skip
        entries: list[TitleEntry] = []
        counts: dict[str, Counter[str]] = {f: Counter() for f in FACET_FIELDS}
        years: list[int] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._collection, limit=2000, offset=offset, with_payload=fields
            )
            for r in records:
                payload = r.payload or {}
                entries.append(_title_entry(int(r.id), payload))
                for field in FACET_FIELDS:
                    value = payload.get(field)
                    values = value if isinstance(value, list) else [value]
                    counts[field].update(str(v) for v in values if v)
                if isinstance(payload.get("start_year"), int):
                    years.append(payload["start_year"])
            if offset is None:
                break
        self._titles = TitleIndex(entries)
        self._facets = Facets(
            **{
                f: [FacetValue(value=v, count=n) for v, n in counts[f].most_common() if v]
                for f in FACET_FIELDS
            },
            year_min=min(years, default=None),
            year_max=max(years, default=None),
        )
        log.info("catalog loaded", titles=len(entries))

    @property
    def titles(self) -> TitleIndex:
        if self._titles is None:
            self._load_catalog()
        assert self._titles is not None
        return self._titles

    @property
    def facets(self) -> Facets:
        """Available filter values with counts, for building UI pickers."""
        if self._facets is None:
            self._load_catalog()
        assert self._facets is not None
        return self._facets

    def get_many(self, anime_ids: Sequence[int]) -> list[AnimeHit]:
        records = self._client.retrieve(self._collection, list(anime_ids), with_payload=True)
        found = {int(r.id): AnimeHit.from_payload(r.payload or {}) for r in records}
        missing = [i for i in anime_ids if i not in found]
        if missing:
            raise AnimeNotFoundError(f"unknown anime_id(s): {missing}")
        return [found[i] for i in anime_ids]

    def popular(self, limit: int = 20) -> list[AnimeHit]:
        """Most-followed anime (MAL members), e.g. for a landing page."""
        return self.get_many([e.anime_id for e in self.titles.most_popular(limit)])

    def resolve_title(self, title: str) -> AnimeHit:
        entry = self.titles.resolve(title)
        if entry is None:
            raise AnimeNotFoundError(f"no anime matches title {title!r}")
        return self.get_many([entry.anime_id])[0]

    # --- retrieval --------------------------------------------------------------

    def _query(
        self,
        query: models.QueryInterface,
        filters: SearchFilters | None,
        pool: int,
        exclude_ids: Sequence[int] = (),
    ) -> list[AnimeHit]:
        points = self._client.query_points(
            self._collection,
            query=query,
            query_filter=to_qdrant_filter(filters, exclude_ids=list(exclude_ids)),
            limit=pool,
            with_payload=True,
        ).points
        return [AnimeHit.from_payload(p.payload or {}, similarity=p.score) for p in points]

    def similar(
        self, anime_id: int, filters: SearchFilters | None = None, limit: int = 10
    ) -> list[AnimeHit]:
        seeds = self.get_many([anime_id])
        hits = self._query(anime_id, filters, max(RERANK_POOL, limit * OVERFETCH), [anime_id])
        # Other seasons of the seed are "similar" but useless as recommendations.
        return collapse_franchises(
            hybrid_rerank(hits, seeds), limit, exclude_keys={franchise_key(seeds[0].title)}
        )

    def search(
        self, text: str, filters: SearchFilters | None = None, limit: int = 10
    ) -> list[AnimeHit]:
        if not text.strip():
            raise ValueError("search text must not be empty")
        vector = self._embedder.embed_query(text)
        return collapse_franchises(self._query(vector, filters, limit * OVERFETCH), limit)

    def taste_profile(
        self,
        liked: Sequence[int],
        disliked: Sequence[int] = (),
        filters: SearchFilters | None = None,
        limit: int = 10,
        strategy: Strategy = "best_score",
        modifier: str | None = None,
    ) -> list[AnimeHit]:
        """Recommend from examples, optionally steered by text ("like X but funnier").

        best_score (default): rank by closeness to *any* liked vs any disliked item, which
        respects varied tastes. average_vector averages the likes into one point, which for
        diverse likes lands in a bland middle (observed: obscure, low-scored results).
        With a `modifier`, its query embedding joins the positives and sum_scores is used,
        so results must be close to the examples *and* the text, not to either alone.
        """
        if not liked:
            raise ValueError("taste profile needs at least one liked anime")
        seeds = self.get_many([*liked, *disliked])
        positive: list[int | list[float]] = list(liked)
        if modifier and modifier.strip():
            positive.append(self._embedder.embed_query(modifier))
            strategy = "sum_scores"
        query = models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=positive,
                negative=list(disliked) or None,
                strategy=models.RecommendStrategy(strategy),
            )
        )
        pool = max(RERANK_POOL, limit * OVERFETCH)
        hits = self._query(query, filters, pool, exclude_ids=[*liked, *disliked])
        return collapse_franchises(
            hybrid_rerank(hits, seeds[: len(liked)]),
            limit,
            exclude_keys={franchise_key(s.title) for s in seeds},
        )

    # --- explanation ------------------------------------------------------------

    def explain(self, request: str, hits: list[AnimeHit], top_n: int = 5) -> Recommendation:
        """LLM re-rank + grounded reasons. Degrades to plain retrieval if unavailable."""
        if self._explainer is None or not hits:
            return Recommendation(items=hits[:top_n])
        return self._explainer.explain(request, hits, top_n=top_n)

    # --- chat -------------------------------------------------------------------

    def chat(
        self,
        message: str,
        filters: SearchFilters | None = None,
        liked_ids: list[int] | None = None,
        disliked_ids: list[int] | None = None,
        limit: int = 6,
        explain: bool = True,
    ) -> tuple[Recommendation, chat_turn.Understood]:
        """One chat turn; see recommender/chat.py for routing."""
        if not message.strip():
            raise ValueError("message must not be empty")
        intent = self._intent_parser.parse(message, self.facets) if self._intent_parser else None
        return chat_turn.respond(self, message, intent=intent, filters=filters,
                                 liked_ids=liked_ids, disliked_ids=disliked_ids,
                                 limit=limit, explain=explain)  # fmt: skip
