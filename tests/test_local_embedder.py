"""Local (sentence-transformers) provider tests with a fake model: no torch download in CI."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anime_rec.config import Settings
from anime_rec.embeddings.base import TaskType
from anime_rec.embeddings.cache import EmbeddingCache
from anime_rec.embeddings.factory import create_embedder
from anime_rec.embeddings.local import LocalEmbedder, Prompts

MODEL = "BAAI/bge-base-en-v1.5"


class FakeSentenceTransformer:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.seen: list[str] = []
        self.kwargs: dict[str, Any] = {}

    def get_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.seen.extend(texts)
        self.kwargs = kwargs
        rng = np.random.default_rng(len(self.seen))
        vectors = rng.normal(size=(len(texts), self.dim))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_bge_query_prefix_applied_only_to_queries() -> None:
    model = FakeSentenceTransformer()
    embedder = LocalEmbedder(MODEL, model=model)
    embedder.embed(["doc text"], TaskType.DOCUMENT)
    embedder.embed_query("dark thriller")
    assert model.seen == [
        "doc text",
        "Represent this sentence for searching relevant passages: dark thriller",
    ]
    assert model.kwargs["normalize_embeddings"] is True


def test_dimension_comes_from_model() -> None:
    assert LocalEmbedder(MODEL, model=FakeSentenceTransformer(dim=384)).dim == 384


def test_unknown_model_embeds_raw_text() -> None:
    model = FakeSentenceTransformer()
    LocalEmbedder("someone/unknown-model", model=model).embed_query("q")
    assert model.seen == ["q"]


def test_cache_reused_and_batches_respected(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    first = FakeSentenceTransformer()
    LocalEmbedder(MODEL, cache, batch_size=2, model=first).embed(list("abcde"), TaskType.DOCUMENT)
    assert first.kwargs["batch_size"] == 2 and len(first.seen) == 5

    second = FakeSentenceTransformer()
    LocalEmbedder(MODEL, cache, model=second).embed(list("abcde"), TaskType.DOCUMENT)
    assert second.seen == []


def test_prompt_change_invalidates_cache(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    LocalEmbedder(MODEL, cache, model=FakeSentenceTransformer()).embed_query("q")
    model = FakeSentenceTransformer()
    LocalEmbedder(MODEL, cache, model=model, prompts=Prompts(query="new: ")).embed_query("q")
    assert model.seen == ["new: q"]


def test_factory_selects_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    import anime_rec.embeddings.local as local

    monkeypatch.setattr(local, "load_sentence_transformer", lambda *_: FakeSentenceTransformer())
    settings = Settings(_env_file=None, embedding_provider="local")  # type: ignore[call-arg]
    embedder = create_embedder(settings)
    assert isinstance(embedder, LocalEmbedder) and embedder.model == MODEL
