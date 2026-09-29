"""Shared test doubles: a tiny catalog in in-memory Qdrant + a keyword embedder."""

from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from anime_rec.embeddings.base import Embedder, TaskType
from anime_rec.recommender.explain import Explainer
from anime_rec.recommender.service import RecommenderService

NAME = "anime_test"
VOCAB = ("titan", "wall", "detective", "notebook", "camp", "girls", "boxing", "time")


class KeywordEmbedder(Embedder):
    """Vector = which vocabulary words occur: similarity is predictable in tests."""

    model = "keyword"
    dim = len(VOCAB)
    batch_size = 32

    def __init__(self) -> None:
        super().__init__(cache=None)
        self.queries: list[str] = []

    def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        if task_type is TaskType.QUERY:
            self.queries.extend(texts)
        out = []
        for text in texts:
            v = np.array([float(w in text.lower()) for w in VOCAB]) + 0.01
            out.append((v / np.linalg.norm(v)).tolist())
        return out


def anime(
    anime_id: int,
    title: str,
    synopsis: str,
    *,
    genres: tuple[str, ...] = (),
    themes: tuple[str, ...] = (),
    type_: str = "TV",
    episodes: int | None = 12,
    year: int = 2015,
    score: float = 8.0,
    members: int = 1000,
) -> dict[str, Any]:
    return {
        "anime_id": anime_id, "title": title, "synopsis": synopsis, "type": type_,
        "episodes": episodes, "start_year": year, "end_year": year, "is_ongoing": False,
        "score": score, "rank": anime_id, "popularity": anime_id, "members": members,
        "image_url": f"https://cdn.myanimelist.net/images/anime/{anime_id}.jpg",
        "mal_url": f"https://myanimelist.net/anime/{anime_id}",
        "genres": list(genres), "themes": list(themes), "demographics": ["Shounen"],
        "studios": ["Studio"], "directors": [], "original_creators": [], "main_characters": [],
    }  # fmt: skip


CATALOG = [
    anime(1, "Shingeki no Kyojin", "giants titan attack the wall", genres=("Action",),
          themes=("Gore",), episodes=25, year=2013, score=8.5, members=4_000_000),
    anime(2, "Shingeki no Kyojin Season 2", "titan wall mystery deepens", genres=("Action",),
          year=2017, score=8.5),
    anime(3, "Shingeki no Kyojin Season 3 Part 2", "titan wall basement", genres=("Action",),
          episodes=10, year=2019, score=9.0),
    anime(4, "Kabaneri", "zombie titan wall train", genres=("Action",), year=2016, score=7.3),
    anime(5, "Death Note", "notebook detective battle of wits", genres=("Suspense",),
          themes=("Psychological",), episodes=37, year=2006, score=8.6, members=4_100_000),
    anime(6, "Monster", "surgeon detective hunts killer", genres=("Suspense",),
          themes=("Psychological",), episodes=74, year=2004, score=8.9),
    anime(7, "Yuru Camp", "girls camp in winter", genres=("Slice of Life",), year=2018),
    anime(8, "Hajime no Ippo", "boxing underdog", genres=("Sports",), episodes=None, year=2000),
    anime(9, "Kimi no Na wa.", "time girls swap bodies", type_="Movie", episodes=1, year=2016),
]  # fmt: skip


def make_service(
    embedder: KeywordEmbedder | None = None, explainer: Explainer | None = None
) -> RecommenderService:
    embedder = embedder or KeywordEmbedder()
    client = QdrantClient(":memory:")
    client.create_collection(
        NAME, vectors_config=models.VectorParams(size=len(VOCAB), distance=models.Distance.COSINE)
    )
    vectors = embedder.embed([a["synopsis"] for a in CATALOG], TaskType.DOCUMENT)
    client.upsert(
        NAME,
        points=[
            models.PointStruct(id=a["anime_id"], vector=v, payload=a)
            for a, v in zip(CATALOG, vectors, strict=True)
        ],
    )
    embedder.queries.clear()
    return RecommenderService(client, NAME, embedder, explainer)
