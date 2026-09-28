"""Vector store tests against Qdrant's in-memory mode (real client logic, no server)."""

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient, models

from anime_rec.ingestion.clean import RawTables, clean_anime
from anime_rec.vectorstore.qdrant import (
    CollectionMismatchError,
    delete_stale,
    ensure_collection,
    point_id,
    to_payload,
    upsert_anime,
)

DIM = 8
NAME = "anime_test"


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def clean(raw_tables: RawTables) -> pd.DataFrame:
    return clean_anime(raw_tables)


def vectors_for(df: pd.DataFrame) -> list[list[float]]:
    rng = np.random.default_rng(0)
    return [list(v / np.linalg.norm(v)) for v in rng.normal(size=(len(df), DIM))]


def index(client: QdrantClient, df: pd.DataFrame, model: str = "m") -> None:
    ensure_collection(client, NAME, DIM, model=model)
    upsert_anime(client, NAME, df, vectors_for(df), template="full", model=model)


def test_point_id_is_the_mal_id() -> None:
    assert point_id(1535) == 1535
    assert point_id(np.int64(21)) == 21
    with pytest.raises(ValueError):
        point_id(0)


def test_reindexing_is_idempotent(client: QdrantClient, clean: pd.DataFrame) -> None:
    index(client, clean)
    index(client, clean)
    assert client.count(NAME).count == len(clean)
    ids = sorted(int(p.id) for p in client.scroll(NAME, limit=100)[0])
    assert ids == sorted(clean["anime_id"].tolist())


def test_reindex_overwrites_payload(client: QdrantClient, clean: pd.DataFrame) -> None:
    index(client, clean)
    index(client, clean.assign(score=1.23))
    point = client.retrieve(NAME, [1535])[0]
    assert point.payload is not None and point.payload["score"] == 1.23


def test_stale_points_removed(client: QdrantClient, clean: pd.DataFrame) -> None:
    index(client, clean)
    remaining = clean[clean["anime_id"] != 99]
    removed = delete_stale(client, NAME, set(remaining["anime_id"].tolist()))
    assert removed == 1
    assert client.count(NAME).count == len(remaining)


def test_dimension_mismatch_is_refused(client: QdrantClient) -> None:
    ensure_collection(client, NAME, DIM)
    with pytest.raises(CollectionMismatchError, match="--recreate"):
        ensure_collection(client, NAME, DIM * 2)
    ensure_collection(client, NAME, DIM * 2, recreate=True)
    params = client.get_collection(NAME).config.params.vectors
    assert isinstance(params, models.VectorParams) and params.size == DIM * 2


def test_payload_is_json_native(clean: pd.DataFrame) -> None:
    row = clean.set_index("anime_id", drop=False).loc[21].to_dict()
    payload = to_payload(row, template="full", model="m")
    assert payload["episodes"] is None  # NA -> null, so "max episodes" filters exclude it
    assert type(payload["anime_id"]) is int and type(payload["score"]) is float
    assert payload["genres"] == ["Action", "Slice of Life"]
    assert payload["embedding_template"] == "full"


def test_filters_work_on_indexed_payload(client: QdrantClient, clean: pd.DataFrame) -> None:
    index(client, clean)
    hits = client.query_points(
        NAME,
        query=vectors_for(clean)[0],
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key="genres", match=models.MatchAny(any=["Suspense"])),
                models.FieldCondition(key="episodes", range=models.Range(lte=50)),
            ]
        ),
        limit=10,
    ).points
    assert [h.id for h in hits] == [1535]


def test_upsert_rejects_length_mismatch(client: QdrantClient, clean: pd.DataFrame) -> None:
    ensure_collection(client, NAME, DIM)
    with pytest.raises(ValueError, match="rows but"):
        upsert_anime(client, NAME, clean, vectors_for(clean)[:1], template="full", model="m")


def test_same_dimension_different_model_is_refused(
    client: QdrantClient, clean: pd.DataFrame
) -> None:
    index(client, clean, model="gemini-embedding-001")
    with pytest.raises(CollectionMismatchError, match="holds vectors from"):
        ensure_collection(client, NAME, DIM, model="BAAI/bge-base-en-v1.5")
    index(client, clean, model="gemini-embedding-001")  # same model: fine
