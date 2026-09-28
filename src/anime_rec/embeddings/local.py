"""Local Hugging Face embedding provider (sentence-transformers), CPU or GPU.

No API, no quota: the whole corpus embeds in minutes. Retrieval-tuned open models expect
an instruction prefix on queries (and some on documents); the prompts are part of the
cache namespace, so changing them can never serve stale vectors.
"""

from dataclasses import dataclass
from typing import Any

from anime_rec.embeddings.base import Embedder, TaskType
from anime_rec.embeddings.cache import EmbeddingCache
from anime_rec.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Prompts:
    query: str = ""
    document: str = ""


_BGE = Prompts(query="Represent this sentence for searching relevant passages: ")
# Prompts as documented on each model card.
KNOWN_PROMPTS: dict[str, Prompts] = {
    "BAAI/bge-small-en-v1.5": _BGE,
    "BAAI/bge-base-en-v1.5": _BGE,
    "BAAI/bge-large-en-v1.5": _BGE,
    "google/embeddinggemma-300m": Prompts(
        query="task: search result | query: ", document="title: none | text: "
    ),
    "intfloat/e5-base-v2": Prompts(query="query: ", document="passage: "),
    "intfloat/e5-large-v2": Prompts(query="query: ", document="passage: "),
}


def load_sentence_transformer(model_name: str, device: str) -> Any:
    # Imported lazily: torch is heavy and only this provider needs it.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=None if device == "auto" else device)


class LocalEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        cache: EmbeddingCache | None = None,
        *,
        device: str = "auto",
        batch_size: int = 64,
        model: Any | None = None,
        prompts: Prompts | None = None,
    ) -> None:
        super().__init__(cache)
        self.model = model_name
        self.batch_size = batch_size
        self._model = model if model is not None else load_sentence_transformer(model_name, device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        if prompts is None:
            prompts = KNOWN_PROMPTS.get(model_name)
            if prompts is None:
                log.warning("no known prompts for model; embedding raw text", model=model_name)
                prompts = Prompts()
        self._prompts = prompts
        log.info("loaded local embedding model", model=model_name, dim=self.dim,
                 device=str(getattr(self._model, "device", device)))  # fmt: skip

    @property
    def cache_namespace(self) -> str:
        return f"{self.model}|q={self._prompts.query}|d={self._prompts.document}"

    def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        prefix = self._prompts.query if task_type is TaskType.QUERY else self._prompts.document
        vectors = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vectors]
