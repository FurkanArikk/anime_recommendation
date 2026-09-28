"""Index stage: processed parquet + cached vectors -> Qdrant (idempotent sync).

Vectors are read from the embedding cache only: indexing never calls the embedding API, so
it is fast, free and repeatable. By default every anime must be embedded; `partial=True`
indexes just the embedded subset (free-tier daily resume) and still syncs exactly.
"""

from anime_rec.config import Settings
from anime_rec.embeddings.cache import EmbeddingCache
from anime_rec.embeddings.factory import create_embedder
from anime_rec.embeddings.pipeline import cached_corpus, embed_corpus
from anime_rec.ingestion.pipeline import load_processed
from anime_rec.log import get_logger
from anime_rec.vectorstore.qdrant import (
    delete_stale,
    ensure_collection,
    get_client,
    point_id,
    upsert_anime,
)

log = get_logger(__name__)


def run_index(settings: Settings, *, recreate: bool = False, partial: bool = False) -> None:
    template = settings.document_template
    df = load_processed(settings)
    rows_in_dataset = len(df)

    cache = EmbeddingCache(settings.embedding_cache_path)
    try:
        embedder = create_embedder(settings, cache=cache)
        if partial:
            df, vectors = cached_corpus(df, embedder, template)
        else:
            vectors = embed_corpus(df, embedder, template, cache_only=True)
    finally:
        cache.close()

    client = get_client(settings)
    name = settings.qdrant_collection
    ensure_collection(client, name, embedder.dim, model=embedder.model, recreate=recreate)
    upserted = upsert_anime(client, name, df, vectors, template=template, model=embedder.model)
    removed = delete_stale(client, name, {point_id(i) for i in df["anime_id"]})

    total = client.count(name, exact=True).count
    if total != len(df):
        raise RuntimeError(f"collection has {total} points, expected {len(df)}")
    log.info(
        "index stage complete",
        collection=name,
        upserted=upserted,
        stale_removed=removed,
        points=total,
        coverage=f"{total}/{rows_in_dataset}",
        template=template,
        model=embedder.model,
    )
