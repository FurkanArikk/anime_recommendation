"""Gemini API embedding provider: throttling, retries and quota handling.

Caching and batching come from `Embedder`. Gemini-specific facts (verified against the API):
batches of at most 100, truncated (non-3072) vectors are not unit length, and the quota
counts every text in a batch as one request.
"""

import time
from collections.abc import Callable
from typing import Any

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
from anime_rec.embeddings.base import Embedder, TaskType, l2_normalize
from anime_rec.embeddings.cache import EmbeddingCache
from anime_rec.log import get_logger

log = get_logger(__name__)

MAX_BATCH = 100  # API limit, verified: 101 inputs -> 400 INVALID_ARGUMENT
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


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


class GeminiEmbedder(Embedder):
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
        super().__init__(cache)
        self._client = client
        self.model = settings.gemini_embedding_model
        self.dim = settings.embedding_dim
        self.batch_size = min(settings.embedding_batch_size, MAX_BATCH)
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

    def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        try:
            return self._call(texts, task_type)
        except errors.APIError as exc:
            if is_daily_quota(exc):
                raise DailyQuotaExceededError(
                    "daily embedding quota exhausted; progress is cached, "
                    "rerun the same command after the quota resets"
                ) from exc
            raise
