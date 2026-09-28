"""FastAPI application. Phase 1: health check only; endpoints arrive in phase 6."""

from fastapi import FastAPI

from anime_rec import __version__
from anime_rec.log import configure_logging

configure_logging()

app = FastAPI(title="Anime Recommender API", version=__version__)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
