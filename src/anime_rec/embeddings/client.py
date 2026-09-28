"""Gemini embedding client with caching, batching, throttling and retries.

Task types matter: Gemini embeds documents and queries into asymmetric spaces tuned for
retrieval, so the index uses RETRIEVAL_DOCUMENT and user queries use RETRIEVAL_QUERY.
"""

import math
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors, types
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from anime_rec.config import Settings
from anime_rec.embeddings.cache import EmbeddingCache, cache_key
from anime_rec.log import get_logger

log = get_logger(__name__)

MAX_BATCH = 100  # API limit, verified: 101 inputs -> 400 INVALID_ARGUMENT
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class TaskType(StrEnum):
    DOCUMENT = "RETRIEVAL_DOCUMENT"
    QUERY = "RETRIEVAL_QUERY"


class Embedder(Protocol):
    """What the rest of the system depends on; tests substitute a fake."""

    dim: int

    def embed(self, texts: Sequence[str], task_type: TaskType) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class CacheMissError(RuntimeError):
    pass


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Truncated (non-3072) Gemini vectors are not unit length (measured norm ~0.59 at 768)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return [x / norm for x in vector]


class DailyQuotaExceededError(RuntimeError):
    """Retrying won't help until the quota resets; progress so far is in the cache."""


def _error_details(exc: BaseException) -> list[dict[str, Any]]:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        inner = details.get("error", {}).get("details", [])
        return [d for d in inner if isinstance(d, dict)]
    return []


def server_retry_delay(exc: BaseException) -> float | None:
    """The 429 body carries google.rpc.RetryInfo, e.g. {"retryDelay": "45s"}."""
    for detail in _error_details(exc):
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def is_daily_quota(exc: BaseException) -> bool:
    for detail in _error_details(exc):
        for violation in detail.get("violations", []):
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_STATUS and not is_daily_quota(exc)
    # Network-level failures (timeouts, connection resets) from the SDK's httpx transport.
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "embedding request failed, backing off",
        attempt=state.attempt_number,
        sleep_s=round(state.next_action.sleep, 1) if state.next_action else None,
        error=str(exc)[:200],
    )


class Throttle:
    """Client-side pacing to at most `per_minute` units, spread evenly.

    Gemini's embedding quota counts every *text* in a batch as one request (verified: one
    100-text batch exhausts the free tier's 100/min), so callers pass the batch size as the
    cost. Staying under the quota is much cheaper than hitting 429s and backing off.
    """

    def __init__(
        self,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / per_minute
        self._clock, self._sleep = clock, sleep
        self._next_at = 0.0

    def wait(self, cost: int = 1) -> None:
        now = self._clock()
        if now < self._next_at:
            self._sleep(self._next_at - now)
            now = self._next_at
        self._next_at = now + cost * self._interval


class GeminiEmbedder:
    def __init__(
        self,
        settings: Settings,
        cache: EmbeddingCache | None = None,
        client: Any | None = None,
        throttle: Throttle | None = None,
        retry_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is None:
            if settings.gemini_api_key is None:
                raise RuntimeError("GEMINI_API_KEY is not set; add it to .env")
            client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self._client = client
        self._cache = cache
        self.model = settings.gemini_embedding_model
        self.dim = settings.embedding_dim
        self._batch_size = min(settings.embedding_batch_size, MAX_BATCH)
        self._throttle = throttle or Throttle(settings.embedding_texts_per_minute)
        backoff = wait_exponential_jitter(initial=2, max=90)

        def wait(state: RetryCallState) -> float:
            # Prefer the server's own RetryInfo; fall back to exponential backoff with jitter.
            exc = state.outcome.exception() if state.outcome else None
            server_delay = server_retry_delay(exc) if exc else None
            return server_delay + 1.0 if server_delay is not None else backoff(state)

        self._call = retry(
            retry=retry_if_exception(is_retryable),
            wait=wait,
            stop=stop_after_attempt(settings.embedding_max_retries + 1),
            before_sleep=_log_retry,
            sleep=retry_sleep,
            reraise=True,
        )(self._request)

    def _request(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        self._throttle.wait(cost=len(texts))
        contents: list[Any] = list(texts)  # SDK's union type rejects list[str] (invariance)
        response = self._client.models.embed_content(
            model=self.model,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type=task_type.value, output_dimensionality=self.dim
            ),
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise RuntimeError(f"got {len(embeddings)} embeddings for {len(texts)} inputs")
        vectors: list[list[float]] = []
        for e in embeddings:
            if e.values is None or len(e.values) != self.dim:
                raise RuntimeError(f"embedding missing or not {self.dim}-dimensional")
            vectors.append(l2_normalize(e.values))
        return vectors

    def lookup(self, texts: Sequence[str], task_type: TaskType) -> list[list[float] | None]:
        """Cached vectors in order, None where missing. Never calls the API."""
        keys = [cache_key(self.model, task_type.value, self.dim, t) for t in texts]
        found = self._cache.get_many(keys) if self._cache is not None else {}
        return [found.get(k) for k in keys]

    def embed(
        self, texts: Sequence[str], task_type: TaskType, *, cache_only: bool = False
    ) -> list[list[float]]:
        """Embed texts in order. Cached vectors are reused; only misses hit the API.
        With `cache_only`, a miss raises instead of calling the API."""
        keys = [cache_key(self.model, task_type.value, self.dim, t) for t in texts]
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
                task_type=task_type.value,
                total=len(texts),
                cached=len(texts) - len(missing),
                to_embed=len(missing),
                batches=math.ceil(len(missing) / self._batch_size),
            )
        items = list(missing.items())
        for start in range(0, len(items), self._batch_size):
            batch = items[start : start + self._batch_size]
            try:
                vectors = self._call([t for _, t in batch], task_type)
            except errors.APIError as exc:
                if is_daily_quota(exc):
                    raise DailyQuotaExceededError(
                        f"daily embedding quota exhausted after {start} new vectors; "
                        "progress is cached, rerun the same command after the quota resets"
                    ) from exc
                raise
            new = {k: v for (k, _), v in zip(batch, vectors, strict=True)}
            if self._cache is not None:
                # Commit per batch: an interrupted run resumes from here.
                self._cache.put_many(new, model=self.model, task_type=task_type.value, dim=self.dim)
            found.update(new)
            done = min(start + self._batch_size, len(items))
            if len(items) > self._batch_size:
                log.info("embedded batch", done=done, total=len(items))
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], TaskType.QUERY)[0]
