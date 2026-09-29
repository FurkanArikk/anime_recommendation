"""Wire the recommender from settings (used by the API and CLI)."""

from anime_rec.config import Settings
from anime_rec.embeddings.cache import EmbeddingCache
from anime_rec.embeddings.factory import create_embedder
from anime_rec.log import get_logger
from anime_rec.recommender.explain import Explainer
from anime_rec.recommender.service import RecommenderService
from anime_rec.vectorstore.qdrant import get_client

log = get_logger(__name__)


def build_service(settings: Settings) -> RecommenderService:
    # Query embeddings are cached too: repeated searches cost nothing.
    embedder = create_embedder(settings, cache=EmbeddingCache(settings.embedding_cache_path))
    explainer = None
    if settings.gemini_api_key is not None:
        explainer = Explainer(settings)
    else:
        log.warning("GEMINI_API_KEY not set: recommendations will not be explained")
    return RecommenderService(get_client(settings), settings.qdrant_collection, embedder, explainer)
