"""Enrich stage: fetch MyAnimeList metadata the scraped dataset lacks, via the Jikan API.

Adds English/Japanese titles and synonyms (so "Attack on Titan" resolves), the complete
MAL genres/themes/demographics (the scrape misses a main genre for ~31% of anime), and
source material, age rating and season.

Jikan (https://jikan.moe) is free and unofficial: ~60 requests/min, and it returns 504
whenever MyAnimeList is slow. So responses are cached one JSON file per anime under
data/raw/jikan/, failures are retried with backoff and otherwise skipped, and every run
only fetches what is missing. Enrichment is optional: ingest works without it.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from anime_rec.config import Settings
from anime_rec.log import get_logger
from anime_rec.ratelimit import Throttle

log = get_logger(__name__)

JIKAN_URL = "https://api.jikan.moe/v4/anime/{anime_id}"
REQUESTS_PER_MINUTE = 50  # Jikan allows 60/min and 3/s; stay under both
MAX_ATTEMPTS = 5
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
NOT_FOUND = {"_not_found": True}
# Circuit breaker: this many failures in a row means Jikan/MAL is down, not one bad id.
MAX_CONSECUTIVE_FAILURES = 10


class RetryableJikanError(RuntimeError):
    pass


class JikanUnavailableError(RuntimeError):
    """Raised by the circuit breaker; everything fetched so far is cached."""


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, (RetryableJikanError, httpx.TransportError))


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.debug("jikan retry", attempt=state.attempt_number, error=str(exc)[:120])


class JikanCache:
    """One JSON file per anime: inspectable, append-only, trivially resumable."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, anime_id: int) -> Path:
        return self.directory / f"{anime_id}.json"

    def has(self, anime_id: int) -> bool:
        return self.path(anime_id).exists()

    def put(self, anime_id: int, data: dict[str, Any]) -> None:
        tmp = self.path(anime_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path(anime_id))  # atomic: never a half-written file

    def load_all(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for file in self.directory.glob("*.json"):
            data = json.loads(file.read_text(encoding="utf-8"))
            if not data.get("_not_found"):
                out[int(file.stem)] = data
        return out


class JikanClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        throttle: Throttle | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        retry_sleep: Any = None,
    ) -> None:
        self._http = http or httpx.Client(timeout=30, headers={"User-Agent": "anime-rec/0.1"})
        self._throttle = throttle or Throttle(REQUESTS_PER_MINUTE)
        kwargs: dict[str, Any] = {"sleep": retry_sleep} if retry_sleep else {}
        self.fetch = retry(
            retry=retry_if_exception(_retryable),
            wait=wait_exponential_jitter(initial=2, max=60),
            stop=stop_after_attempt(max_attempts),
            before_sleep=_log_retry,
            reraise=True,
            **kwargs,
        )(self._fetch)

    def _fetch(self, anime_id: int) -> dict[str, Any]:
        """The anime's `data` object, or NOT_FOUND for ids MAL doesn't know."""
        self._throttle.wait()
        response = self._http.get(JIKAN_URL.format(anime_id=anime_id))
        if response.status_code == 404:
            return dict(NOT_FOUND)
        if response.status_code in RETRYABLE_STATUS:
            raise RetryableJikanError(f"HTTP {response.status_code} for anime {anime_id}")
        response.raise_for_status()
        data: dict[str, Any] = response.json()["data"]
        return data


def _names(items: Any) -> list[str]:
    return [i["name"] for i in items or [] if i.get("name")]


def parse_jikan(data: dict[str, Any]) -> dict[str, Any]:
    """Jikan `data` -> the fields we merge into the cleaned dataset."""
    default = data.get("title")
    synonyms = [
        t["title"]
        for t in data.get("titles") or []
        if t.get("type") not in ("Default", "Japanese") and t.get("title") != default
    ]
    season, year = data.get("season"), data.get("year")
    return {
        "title_english": data.get("title_english") or None,
        "title_japanese": data.get("title_japanese") or None,
        "title_synonyms": sorted(set(synonyms)),
        "genres": sorted(set(_names(data.get("genres")) + _names(data.get("explicit_genres")))),
        "themes": sorted(set(_names(data.get("themes")))),
        "demographics": sorted(set(_names(data.get("demographics")))),
        "source": data.get("source") or None,
        "age_rating": data.get("rating") or None,
        "season": f"{season.capitalize()} {year}" if season and year else None,
    }


def load_enrichment(directory: Path) -> pd.DataFrame | None:
    """Parsed enrichment for every cached anime, or None if the stage never ran."""
    if not directory.exists():
        return None
    records = [
        {"anime_id": i, **parse_jikan(d)} for i, d in JikanCache(directory).load_all().items()
    ]
    if not records:
        return None
    return pd.DataFrame(records).set_index("anime_id")


def run_enrich(
    settings: Settings, limit: int | None = None, client: JikanClient | None = None
) -> dict[str, int]:
    # Read ids from the raw CSV so enrich can run before or after ingest.
    raw = pd.read_csv(settings.raw_dir / "anime.csv", usecols=["anime_id", "rank"])
    ids = [int(i) for i in raw.sort_values("rank")["anime_id"].drop_duplicates()]
    if limit:
        ids = ids[:limit]
    cache = JikanCache(settings.jikan_dir)
    todo = [i for i in ids if not cache.has(i)]
    log.info("enrich plan", total=len(ids), cached=len(ids) - len(todo), to_fetch=len(todo),
             eta_min=round(len(todo) * 60 / REQUESTS_PER_MINUTE / 60))  # fmt: skip

    client = client or JikanClient()
    stats = {"fetched": 0, "not_found": 0, "failed": 0}
    consecutive_failures = 0
    for n, anime_id in enumerate(todo, start=1):
        try:
            data = client.fetch(anime_id)
        except (RetryableJikanError, httpx.HTTPError) as exc:
            stats["failed"] += 1  # left uncached: the next run retries it
            consecutive_failures += 1
            log.warning("jikan fetch failed, skipping", anime_id=anime_id, error=str(exc)[:120])
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise JikanUnavailableError(
                    f"{consecutive_failures} consecutive failures: Jikan/MyAnimeList looks down. "
                    f"Progress is cached ({stats['fetched']} fetched this run); rerun later."
                ) from exc
            continue
        consecutive_failures = 0
        cache.put(anime_id, data)
        stats["not_found" if data.get("_not_found") else "fetched"] += 1
        if n % 100 == 0:
            log.info("enrich progress", done=n, total=len(todo), **stats)
    log.info("enrich stage complete", **stats, remaining=stats["failed"])
    return stats
