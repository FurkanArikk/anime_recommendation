"""On-disk embedding cache (SQLite, stdlib only).

The key is sha256(model | task_type | dim | text), so a vector is reused only when *everything*
that determines it is identical: editing a template, switching model, or changing the
dimension produces new keys instead of silently serving stale vectors. SQLite gives atomic
per-batch commits, so a crash or Ctrl-C loses at most the batch in flight.
"""

import hashlib
import sqlite3
import threading
from array import array
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key        TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    task_type  TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""
_SQLITE_MAX_PARAMS = 900  # stay under SQLite's host-parameter limit for IN (...) queries


def cache_key(model: str, task_type: str, dim: int, text: str) -> str:
    # \x1f (unit separator) can't occur in normal text, so fields can't bleed into each other.
    payload = "\x1f".join([model, task_type, str(dim), text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_blob(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _from_blob(blob: bytes) -> list[float]:
    return array("f", blob).tolist()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The API calls this from FastAPI's worker threads: allow cross-thread use and
        # serialise access ourselves (SQLite connections are not safe to share unlocked).
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)

    def get_many(self, keys: Iterable[str]) -> dict[str, list[float]]:
        keys = list(dict.fromkeys(keys))
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), _SQLITE_MAX_PARAMS):
            chunk = keys[i : i + _SQLITE_MAX_PARAMS]
            placeholders = ",".join("?" * len(chunk))
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", chunk
                ).fetchall()
            found.update((key, _from_blob(blob)) for key, blob in rows)
        return found

    def put_many(
        self, vectors: Mapping[str, Sequence[float]], *, model: str, task_type: str, dim: int
    ) -> None:
        with self._lock, self._conn:  # one transaction per call = one atomic commit per batch
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, model, task_type, dim, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                [(k, model, task_type, dim, _to_blob(v)) for k, v in vectors.items()],
            )

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
