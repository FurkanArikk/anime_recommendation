"""Qdrant collection management and idempotent indexing.

Point IDs are the MAL anime IDs: stable, unique, and meaningful. Re-running the index stage
overwrites points in place instead of duplicating them, and a sync step removes points for
anime that disappeared from the dataset, so the collection always mirrors the parquet.
"""

from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from anime_rec.config import Settings
from anime_rec.log import get_logger

log = get_logger(__name__)

# Fields the recommender filters on. Indexed fields make filtered HNSW search fast and let
# Qdrant plan filters well; unindexed payload is still stored for display.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "score": models.PayloadSchemaType.FLOAT,
    "start_year": models.PayloadSchemaType.INTEGER,
    "episodes": models.PayloadSchemaType.INTEGER,
    "members": models.PayloadSchemaType.INTEGER,
    "type": models.PayloadSchemaType.KEYWORD,
    "genres": models.PayloadSchemaType.KEYWORD,
    "themes": models.PayloadSchemaType.KEYWORD,
    "demographics": models.PayloadSchemaType.KEYWORD,
    "studios": models.PayloadSchemaType.KEYWORD,
    "is_ongoing": models.PayloadSchemaType.BOOL,
}

UPSERT_BATCH = 256


class CollectionMismatchError(RuntimeError):
    pass


def get_client(settings: Settings) -> QdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=60)


def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    *,
    model: str | None = None,
    recreate: bool = False,
) -> None:
    """Create the collection (cosine, `dim`) and payload indexes if missing.

    An existing collection built differently is an error, not something to silently write
    into. A dimension check alone is not enough: two models can both produce 768-d vectors
    that live in unrelated spaces, so the embedding model recorded on points is checked too.
    """
    if recreate and client.collection_exists(name):
        log.warning("dropping collection for recreate", collection=name)
        client.delete_collection(name)

    if client.collection_exists(name):
        params = client.get_collection(name).config.params.vectors
        if not isinstance(params, models.VectorParams) or (
            params.size != dim or params.distance != models.Distance.COSINE
        ):
            raise CollectionMismatchError(
                f"collection {name!r} has vectors {params}, expected size={dim} cosine; "
                "re-run with --recreate"
            )
        existing = indexed_model(client, name) if model is not None else None
        if existing is not None and existing != model:
            raise CollectionMismatchError(
                f"collection {name!r} holds vectors from {existing!r}, not {model!r}; "
                "re-run with --recreate"
            )
    else:
        client.create_collection(
            name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )
        log.info("created collection", collection=name, dim=dim, distance="cosine")

    # Idempotent: creating an index that already exists is a no-op server-side.
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=schema)


def indexed_model(client: QdrantClient, name: str) -> str | None:
    """Embedding model recorded on an existing point (None for an empty collection)."""
    records, _ = client.scroll(name, limit=1, with_payload=["embedding_model"])
    if not records or not records[0].payload:
        return None
    value = records[0].payload.get("embedding_model")
    return str(value) if value is not None else None


def _native(value: Any) -> Any:
    """pandas/numpy values -> JSON-safe Python (NA -> None, np.int64 -> int, arrays -> lists)."""
    if isinstance(value, Mapping):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(v) for v in value]
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_payload(row: Mapping[Hashable, Any], *, template: str, model: str) -> dict[str, Any]:
    payload = {str(k): _native(v) for k, v in row.items()}
    # Provenance: which document template / model produced this point's vector.
    payload["embedding_template"] = template
    payload["embedding_model"] = model
    return payload


def point_id(anime_id: Any) -> int:
    """Deterministic point ID: the MAL anime ID itself (Qdrant accepts unsigned ints)."""
    value = int(anime_id)
    if value <= 0:
        raise ValueError(f"anime_id must be positive, got {anime_id!r}")
    return value


def _batches(points: Sequence[models.PointStruct], size: int) -> Iterator[list[models.PointStruct]]:
    for i in range(0, len(points), size):
        yield list(points[i : i + size])


def upsert_anime(
    client: QdrantClient,
    name: str,
    df: pd.DataFrame,
    vectors: Sequence[Sequence[float]],
    *,
    template: str,
    model: str,
) -> int:
    if len(df) != len(vectors):
        raise ValueError(f"{len(df)} rows but {len(vectors)} vectors")
    points = [
        models.PointStruct(
            id=point_id(row["anime_id"]),
            vector=list(vector),
            payload=to_payload(row, template=template, model=model),
        )
        for row, vector in zip(df.to_dict("records"), vectors, strict=True)
    ]
    for batch in _batches(points, UPSERT_BATCH):
        client.upsert(name, points=batch, wait=True)
    return len(points)


def delete_stale(client: QdrantClient, name: str, keep_ids: set[int]) -> int:
    """Remove points whose anime is no longer in the dataset."""
    stale: list[int] = []
    offset: Any = None
    while True:
        records, offset = client.scroll(
            name, limit=1000, offset=offset, with_payload=False, with_vectors=False
        )
        stale.extend(int(r.id) for r in records if int(r.id) not in keep_ids)
        if offset is None:
            break
    if stale:
        client.delete(name, points_selector=models.PointIdsList(points=list(stale)), wait=True)
    return len(stale)
