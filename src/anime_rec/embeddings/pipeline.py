"""Embed stage: processed parquet -> documents -> vectors in the on-disk cache.

The cache *is* this stage's output. The index stage re-derives the same documents and
reads vectors from the cache only, so indexing can never trigger surprise API calls.

Free-tier note: Gemini allows 1,000 embedded texts per day, shared with query embeddings
from the running app. `max_new` caps how many new documents one run may embed so the app
keeps some quota for searches; rows are processed in rank order, so a partially embedded
corpus is always the top-N anime.
"""

import pandas as pd

from anime_rec.config import Settings
from anime_rec.embeddings.cache import EmbeddingCache
from anime_rec.embeddings.client import GeminiEmbedder, TaskType
from anime_rec.embeddings.documents import build_documents
from anime_rec.ingestion.pipeline import load_processed
from anime_rec.log import get_logger

log = get_logger(__name__)


def embed_corpus(
    df: pd.DataFrame,
    embedder: GeminiEmbedder,
    template: str,
    *,
    cache_only: bool = False,
) -> list[list[float]]:
    documents = build_documents(df, template)
    return embedder.embed(documents, TaskType.DOCUMENT, cache_only=cache_only)


def cached_corpus(
    df: pd.DataFrame, embedder: GeminiEmbedder, template: str
) -> tuple[pd.DataFrame, list[list[float]]]:
    """The subset of `df` whose document vectors are already cached, with those vectors."""
    looked_up = embedder.lookup(build_documents(df, template), TaskType.DOCUMENT)
    mask = [v is not None for v in looked_up]
    return df[mask], [v for v in looked_up if v is not None]


def run_embed(
    settings: Settings,
    template: str | None = None,
    limit: int | None = None,
    max_new: int | None = None,
) -> None:
    template = template or settings.document_template
    df = load_processed(settings)
    if limit:
        df = df.head(limit)  # sorted by rank, so a trial run embeds the best-known titles
    cache = EmbeddingCache(settings.embedding_cache_path)
    try:
        embedder = GeminiEmbedder(settings, cache=cache)
        documents = build_documents(df, template)
        pending = [
            d
            for d, v in zip(documents, embedder.lookup(documents, TaskType.DOCUMENT), strict=True)
            if v is None
        ]
        todo = pending if max_new is None else pending[:max_new]
        log.info(
            "embed plan",
            template=template,
            documents=len(documents),
            already_cached=len(documents) - len(pending),
            this_run=len(todo),
            left_after_run=len(pending) - len(todo),
        )
        embedder.embed(todo, TaskType.DOCUMENT)
        log.info("embed stage complete", remaining=len(pending) - len(todo), dim=embedder.dim)
    finally:
        cache.close()
