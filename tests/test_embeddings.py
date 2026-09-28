"""Embedding client tests with the Gemini SDK replaced by an in-memory fake (no network)."""

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from google.genai import errors

from anime_rec.config import Settings
from anime_rec.embeddings.cache import EmbeddingCache, cache_key
from anime_rec.embeddings.client import (
    CacheMissError,
    DailyQuotaExceededError,
    GeminiEmbedder,
    TaskType,
    Throttle,
    l2_normalize,
)
from anime_rec.embeddings.documents import build_documents
from anime_rec.embeddings.pipeline import cached_corpus

DIM = 768


def fake_vector(text: str, task_type: str) -> list[float]:
    """Deterministic, deliberately *not* unit-length, like real truncated Gemini output."""
    seed = hashlib.sha256(f"{task_type}:{text}".encode()).digest()
    return [(seed[i % 32] - 128) / 400 for i in range(DIM)]


class FakeModels:
    def __init__(self, failures: list[Exception | None] | None = None) -> None:
        """`failures[i]` is raised on call i (None = succeed); later calls succeed."""
        self.calls: list[dict[str, Any]] = []
        self._failures = list(failures or [])

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        failure = self._failures.pop(0) if self._failures else None
        if failure is not None:
            raise failure
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=fake_vector(t, config.task_type)) for t in contents]
        )


def api_error(code: int, details: list[dict[str, Any]] | None = None) -> errors.APIError:
    cls = errors.ClientError if code < 500 else errors.ServerError
    body = {"error": {"code": code, "message": "boom", "status": "X", "details": details or []}}
    return cls(code, body)


def quota_error(quota_id: str, retry_delay: str = "45s") -> errors.APIError:
    """Shape copied from a real free-tier 429 response."""
    return api_error(
        429,
        [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": quota_id, "quotaValue": "100"}],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        embedding_batch_size=2,
        embedding_texts_per_minute=600,
        embedding_max_retries=3,
    )


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "emb.sqlite")


def make_embedder(
    settings: Settings,
    cache: EmbeddingCache | None,
    models: FakeModels,
    retry_sleeps: list[float] | None = None,
) -> GeminiEmbedder:
    no_wait = Throttle(600, sleep=lambda _: None)
    sleeps = retry_sleeps if retry_sleeps is not None else []
    return GeminiEmbedder(
        settings,
        cache=cache,
        client=SimpleNamespace(models=models),
        throttle=no_wait,
        retry_sleep=sleeps.append,
    )


def test_batches_normalizes_and_preserves_order(settings: Settings) -> None:
    models = FakeModels()
    texts = ["a", "b", "c", "d", "e"]
    vectors = make_embedder(settings, None, models).embed(texts, TaskType.DOCUMENT)

    assert [len(c["contents"]) for c in models.calls] == [2, 2, 1]
    assert all(c["config"].output_dimensionality == DIM for c in models.calls)
    assert all(math.isclose(math.hypot(*v), 1.0, rel_tol=1e-6) for v in vectors)
    assert vectors[3] == l2_normalize(fake_vector("d", "RETRIEVAL_DOCUMENT"))


def test_query_uses_retrieval_query_task(settings: Settings) -> None:
    models = FakeModels()
    make_embedder(settings, None, models).embed_query("dark thriller")
    assert models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_second_run_is_fully_cached(settings: Settings, cache: EmbeddingCache) -> None:
    texts = ["a", "b", "c"]
    first = make_embedder(settings, cache, FakeModels()).embed(texts, TaskType.DOCUMENT)

    models = FakeModels()
    second = make_embedder(settings, cache, models).embed(texts, TaskType.DOCUMENT)
    assert models.calls == []
    for cached, original in zip(second, first, strict=True):
        assert cached == pytest.approx(original, rel=1e-6)  # float32 round-trip


def test_interrupted_run_resumes_without_reembedding(
    settings: Settings, cache: EmbeddingCache
) -> None:
    # Batch 1 succeeds, batch 2 fails hard: batch 1 must already be committed to the cache.
    crashing = FakeModels(failures=[None, api_error(400)])
    with pytest.raises(errors.ClientError):
        make_embedder(settings, cache, crashing).embed(["a", "b", "c", "d"], TaskType.DOCUMENT)

    models = FakeModels()
    make_embedder(settings, cache, models).embed(["a", "b", "c", "d"], TaskType.DOCUMENT)
    assert [c["contents"] for c in models.calls] == [["c", "d"]]


def test_duplicate_texts_embedded_once(settings: Settings) -> None:
    models = FakeModels()
    vectors = make_embedder(settings, None, models).embed(["x", "x", "y"], TaskType.DOCUMENT)
    assert sum(len(c["contents"]) for c in models.calls) == 2
    assert vectors[0] == vectors[1]


def test_cache_key_depends_on_everything_that_changes_the_vector() -> None:
    base = cache_key("m", "RETRIEVAL_DOCUMENT", 768, "text")
    assert base == cache_key("m", "RETRIEVAL_DOCUMENT", 768, "text")
    assert base != cache_key("m2", "RETRIEVAL_DOCUMENT", 768, "text")
    assert base != cache_key("m", "RETRIEVAL_QUERY", 768, "text")
    assert base != cache_key("m", "RETRIEVAL_DOCUMENT", 1536, "text")
    assert base != cache_key("m", "RETRIEVAL_DOCUMENT", 768, "text ")


@pytest.mark.parametrize("code", [429, 500, 503])
def test_retries_transient_errors(settings: Settings, code: int) -> None:
    models = FakeModels(failures=[api_error(code), api_error(code)])
    vectors = make_embedder(settings, None, models).embed(["a"], TaskType.DOCUMENT)
    assert len(models.calls) == 3 and len(vectors) == 1


def test_gives_up_after_max_retries(settings: Settings) -> None:
    models = FakeModels(failures=[api_error(429) for _ in range(10)])
    with pytest.raises(errors.ClientError):
        make_embedder(settings, None, models).embed(["a"], TaskType.DOCUMENT)
    assert len(models.calls) == settings.embedding_max_retries + 1


def test_client_errors_are_not_retried(settings: Settings) -> None:
    models = FakeModels(failures=[api_error(400)])
    with pytest.raises(errors.ClientError):
        make_embedder(settings, None, models).embed(["a"], TaskType.DOCUMENT)
    assert len(models.calls) == 1


def test_cache_only_raises_on_miss(settings: Settings, cache: EmbeddingCache) -> None:
    models = FakeModels()
    with pytest.raises(CacheMissError):
        make_embedder(settings, cache, models).embed(["a"], TaskType.DOCUMENT, cache_only=True)
    assert models.calls == []


def test_backoff_honors_server_retry_delay(settings: Settings) -> None:
    sleeps: list[float] = []
    models = FakeModels(failures=[quota_error("EmbedContentRequestsPerMinute-FreeTier", "45s")])
    make_embedder(settings, None, models, retry_sleeps=sleeps).embed(["a"], TaskType.DOCUMENT)
    assert sleeps == [46.0]  # server delay + 1s margin


def test_daily_quota_stops_immediately_and_keeps_progress(
    settings: Settings, cache: EmbeddingCache
) -> None:
    models = FakeModels(failures=[None, quota_error("EmbedContentRequestsPerDay-FreeTier")])
    with pytest.raises(DailyQuotaExceededError, match="rerun"):
        make_embedder(settings, cache, models).embed(["a", "b", "c", "d"], TaskType.DOCUMENT)
    assert len(models.calls) == 2  # no retries against a daily limit
    assert len(cache) == 2  # first batch kept


def test_throttle_charges_by_batch_size() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    throttle = Throttle(per_minute=100, clock=lambda: now[0], sleep=sleep)
    throttle.wait(cost=50)
    throttle.wait(cost=50)  # 50 texts at 100/min -> 30 s later
    assert slept == [30.0]


def test_throttle_spaces_calls() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    throttle = Throttle(per_minute=60, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        throttle.wait()
    assert slept == [1.0, 1.0]


def test_lookup_returns_cached_or_none_without_api_calls(
    settings: Settings, cache: EmbeddingCache
) -> None:
    make_embedder(settings, cache, FakeModels()).embed(["a"], TaskType.DOCUMENT)
    models = FakeModels()
    found = make_embedder(settings, cache, models).lookup(["a", "b"], TaskType.DOCUMENT)
    assert found[0] is not None and found[1] is None
    assert models.calls == []


def test_cached_corpus_returns_embedded_subset_in_order(
    settings: Settings, cache: EmbeddingCache
) -> None:
    df = pd.DataFrame(
        {"anime_id": [1, 2, 3], "title": ["A", "B", "C"], "synopsis": ["x", "y", "z"]}
    )
    embedder = make_embedder(settings, cache, FakeModels())
    docs = build_documents(df, "synopsis_only")
    embedder.embed([docs[0], docs[2]], TaskType.DOCUMENT)  # anime 2 not embedded yet

    subset, vectors = cached_corpus(df, embedder, "synopsis_only")
    assert subset["anime_id"].tolist() == [1, 3]
    assert vectors == embedder.lookup([docs[0], docs[2]], TaskType.DOCUMENT)
