"""Embed stage: processed parquet -> documents -> vectors in the on-disk cache.

The cache *is* this stage's output. The index stage re-derives the same documents and
reads vectors with `cache_only=True`, so indexing can never trigger surprise API calls.
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


def run_embed(settings: Settings, template: str | None = None, limit: int | None = None) -> None:
    template = template or settings.document_template
    df = load_processed(settings)
    if limit:
        df = df.head(limit)  # sorted by rank, so a trial run embeds the best-known titles
    cache = EmbeddingCache(settings.embedding_cache_path)
    try:
        embedder = GeminiEmbedder(settings, cache=cache)
        vectors = embed_corpus(df, embedder, template)
        log.info(
            "embed stage complete",
            template=template,
            documents=len(vectors),
            dim=embedder.dim,
            cache_entries=len(cache),
        )
    finally:
        cache.close()
