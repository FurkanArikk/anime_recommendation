"""Build the configured embedding provider (EMBEDDING_PROVIDER=local|gemini)."""

from anime_rec.config import Settings
from anime_rec.embeddings.base import Embedder
from anime_rec.embeddings.cache import EmbeddingCache


def create_embedder(settings: Settings, cache: EmbeddingCache | None = None) -> Embedder:
    if settings.embedding_provider == "gemini":
        from anime_rec.embeddings.gemini import GeminiEmbedder

        return GeminiEmbedder(settings, cache=cache)

    from anime_rec.embeddings.local import LocalEmbedder

    return LocalEmbedder(
        settings.local_embedding_model,
        cache=cache,
        device=settings.embedding_device,
        batch_size=settings.local_batch_size,
    )
