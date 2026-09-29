"""FastAPI backend: a thin HTTP layer over RecommenderService.

The service (embedding model, Qdrant client, title index) is built once at startup via the
lifespan hook. `create_app` takes the factory as an argument so tests inject a fake
service: no model download, no cluster.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from anime_rec import __version__
from anime_rec.api.schemas import (
    AnimeRef,
    ReadyResponse,
    RecommendationResponse,
    RecommendRequest,
    SearchRequest,
    SimilarRequest,
)
from anime_rec.config import Settings, get_settings
from anime_rec.log import configure_logging, get_logger
from anime_rec.recommender.schemas import AnimeHit, Facets
from anime_rec.recommender.service import AnimeNotFoundError, RecommenderService

log = get_logger(__name__)

ServiceFactory = Callable[[Settings], RecommenderService]
CANDIDATES_FOR_EXPLAIN = 2  # retrieve this many x limit so the LLM has room to choose


def _default_factory(settings: Settings) -> RecommenderService:
    from anime_rec.recommender.factory import build_service

    return build_service(settings)


def get_service(request: Request) -> RecommenderService:
    service: RecommenderService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "service is starting up")
    return service


Service = Annotated[RecommenderService, Depends(get_service)]


def _ref(hit: AnimeHit) -> AnimeRef:
    return AnimeRef(
        anime_id=hit.anime_id,
        title=hit.title,
        title_english=hit.title_english,
        image_url=hit.image_url,
        members=hit.members,
    )


def _respond(
    service: RecommenderService,
    hits: list[AnimeHit],
    *,
    explain: bool,
    request_text: str,
    limit: int,
    started: float,
    seeds: list[AnimeHit] | None = None,
) -> RecommendationResponse:
    if explain:
        rec = service.explain(request_text, hits, top_n=limit)
        items, summary, explained = rec.items, rec.summary, rec.explained
    else:
        items, summary, explained = hits[:limit], None, False
    return RecommendationResponse(
        items=items,
        summary=summary,
        explained=explained,
        seeds=[_ref(s) for s in seeds or []],
        took_ms=round((time.perf_counter() - started) * 1000),
    )


def create_app(service_factory: ServiceFactory = _default_factory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        started = time.perf_counter()
        service = service_factory(get_settings())
        service.titles  # noqa: B018 - warm the catalog so the first request is fast
        app.state.service = service
        log.info("api ready", startup_s=round(time.perf_counter() - started, 1))
        yield

    app = FastAPI(
        title="Anime Recommender API",
        version=__version__,
        description="Semantic search, similar-anime and taste-profile recommendations over "
        "the MyAnimeList top 10k, with optional grounded Gemini explanations.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - started) * 1000),
        )
        return response

    @app.exception_handler(AnimeNotFoundError)
    async def not_found(_: Request, exc: AnimeNotFoundError) -> Response:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    # --- meta -------------------------------------------------------------------

    @app.get("/health", tags=["meta"], summary="Liveness: the process is up")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["meta"], summary="Readiness: Qdrant reachable, models loaded")
    def ready(service: Service) -> ReadyResponse:
        try:
            points = service.count()
        except (ResponseHandlingException, UnexpectedResponse, OSError) as exc:
            raise HTTPException(503, f"Qdrant unavailable: {exc}") from exc
        return ReadyResponse(
            status="ready",
            collection=service.collection,
            points=points,
            embedding_model=service.embedding_model,
            chat_models=service.chat_models,
        )

    # --- catalog ----------------------------------------------------------------

    @app.get("/anime/{anime_id}", tags=["catalog"])
    def get_anime(anime_id: int, service: Service) -> AnimeHit:
        return service.get_many([anime_id])[0]

    @app.get("/titles", tags=["catalog"], summary="Title autocomplete (romaji, English, synonyms)")
    def titles(
        service: Service,
        q: Annotated[str, Query(max_length=100)] = "",
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> list[AnimeRef]:
        return [
            AnimeRef(anime_id=e.anime_id, title=e.title, members=e.members, title_english=e.english)
            for e in service.titles.suggest(q, limit)
        ]

    @app.get("/facets", tags=["catalog"], summary="Filter values with counts, for UI pickers")
    def facets(service: Service) -> Facets:
        return service.facets

    # --- recommendations --------------------------------------------------------

    @app.post("/search", tags=["recommend"], summary="Natural-language search")
    def search(body: SearchRequest, service: Service) -> RecommendationResponse:
        started = time.perf_counter()
        pool = body.limit * (CANDIDATES_FOR_EXPLAIN if body.explain else 1)
        try:
            hits = service.search(body.query, body.filters, limit=pool)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return _respond(
            service,
            hits,
            explain=body.explain,
            request_text=body.query,
            limit=body.limit,
            started=started,
        )

    @app.post("/similar", tags=["recommend"], summary="Anime similar to one anime")
    def similar(body: SimilarRequest, service: Service) -> RecommendationResponse:
        started = time.perf_counter()
        seed = (
            service.get_many([body.anime_id])[0]
            if body.anime_id is not None
            else service.resolve_title(body.title or "")
        )
        pool = body.limit * (CANDIDATES_FOR_EXPLAIN if body.explain else 1)
        hits = service.similar(seed.anime_id, body.filters, limit=pool)
        return _respond(
            service,
            hits,
            explain=body.explain,
            request_text=f"Anime similar to {seed.title}: {seed.synopsis or ''}"[:800],
            limit=body.limit,
            started=started,
            seeds=[seed],
        )

    @app.post("/recommend", tags=["recommend"], summary="Taste profile from liked/disliked")
    def recommend(body: RecommendRequest, service: Service) -> RecommendationResponse:
        started = time.perf_counter()
        seeds = service.get_many([*body.liked, *body.disliked])
        liked, disliked = seeds[: len(body.liked)], seeds[len(body.liked) :]
        pool = body.limit * (CANDIDATES_FOR_EXPLAIN if body.explain else 1)
        try:
            hits = service.taste_profile(
                body.liked, body.disliked, body.filters, limit=pool, strategy=body.strategy
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        text = "The user liked: " + ", ".join(s.title for s in liked)
        if disliked:
            text += ". They disliked: " + ", ".join(s.title for s in disliked)
        return _respond(
            service,
            hits,
            explain=body.explain,
            request_text=text,
            limit=body.limit,
            started=started,
            seeds=seeds,
        )

    return app


app = create_app()
