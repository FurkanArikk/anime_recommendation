"""Provider-independent embedding logic: caching, dedup, batching, normalisation.

Providers (Gemini API, local sentence-transformers) only implement `_embed_batch`; the
guarantees that matter for the pipeline (a vector is never computed twice, an interrupted
run resumes where it stopped, indexing can run cache-only) live here, once.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from anime_rec.embeddings.cache import EmbeddingCache, cache_key
from anime_rec.log import get_logger

log = get_logger(__name__)


class TaskType(StrEnum):
    """Retrieval is asymmetric: short queries and long documents are embedded differently
    (Gemini task types; instruction prefixes for open models)."""

    DOCUMENT = "RETRIEVAL_DOCUMENT"
    QUERY = "RETRIEVAL_QUERY"


class CacheMissError(RuntimeError):
    pass


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return [x / norm for x in vector]


class Embedder(ABC):
    #: Model identifier stored with each point, e.g. "BAAI/bge-base-en-v1.5".
    model: str
    dim: int
    batch_size: int

    def __init__(self, cache: EmbeddingCache | None) -> None:
        self._cache = cache

    @property
    def cache_namespace(self) -> str:
        """Everything besides task type, dim and text that determines a vector."""
        return self.model

    @abstractmethod
    def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        """Return one unit-length vector per text, in order."""

    def _keys(self, texts: Sequence[str], task_type: TaskType) -> list[str]:
        return [cache_key(self.cache_namespace, task_type.value, self.dim, t) for t in texts]

    def lookup(self, texts: Sequence[str], task_type: TaskType) -> list[list[float] | None]:
        """Cached vectors in order, None where missing. Never computes anything."""
        keys = self._keys(texts, task_type)
        found = self._cache.get_many(keys) if self._cache is not None else {}
        return [found.get(k) for k in keys]

    def embed(
        self, texts: Sequence[str], task_type: TaskType, *, cache_only: bool = False
    ) -> list[list[float]]:
        """Embed texts in order. Cached vectors are reused; only misses are computed.
        With `cache_only`, a miss raises instead of computing."""
        keys = self._keys(texts, task_type)
        found = self._cache.get_many(keys) if self._cache is not None else {}
        # Deduplicate misses: identical texts are embedded once.
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}

        if missing and cache_only:
            raise CacheMissError(
                f"{len(missing)} of {len(texts)} texts are not in the embedding cache"
            )
        if missing:
            log.info(
                "embedding",
                model=self.model,
                task_type=task_type.value,
                total=len(texts),
                cached=len(texts) - len(missing),
                to_embed=len(missing),
                batches=math.ceil(len(missing) / self.batch_size),
            )
        items = list(missing.items())
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            vectors = self._embed_batch([t for _, t in batch], task_type)
            new = {k: v for (k, _), v in zip(batch, vectors, strict=True)}
            if self._cache is not None:
                # Commit per batch: an interrupted run resumes from here.
                self._cache.put_many(new, model=self.model, task_type=task_type.value, dim=self.dim)
            found.update(new)
            if len(items) > self.batch_size:
                done = min(start + self.batch_size, len(items))
                log.info("embedded batch", done=done, total=len(items))
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], TaskType.QUERY)[0]
