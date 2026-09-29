"""Client-side request pacing shared by API clients (Gemini embeddings, Jikan)."""

import time
from collections.abc import Callable


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
