# Deployment

The production stack is two containers behind one public port:

```
internet ──TLS──▶ reverse proxy (Caddy / Traefik / cloud LB)
                     │ :80
                     ▼
               web (nginx) ── static SPA
                     │ /api/*  (same origin: no CORS)
                     ▼
               api (FastAPI + EmbeddingGemma on CPU) ──▶ Qdrant Cloud
                                                     └──▶ Gemini API
```

## 1. Server

Any Linux VM with Docker (Compose ≥ 2.24) and **≥ 4 GB RAM**: the API holds the
EmbeddingGemma model (~1.2 GB) in memory. 2 vCPU gives about 50–100 ms per query embedding.
No GPU is needed to serve; a GPU only speeds up the one-off batch embedding.

## 2. Configuration

```bash
git clone https://github.com/FurkanArikk/anime_recommendation.git && cd anime_recommendation
make env        # creates .env from .env.example
```

Fill in `.env`:

| Variable | Required | Notes |
|---|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | yes | Qdrant Cloud cluster (include `:6333`) |
| `HF_TOKEN` | yes | read token; accept the `google/embeddinggemma-300m` license first |
| `GEMINI_API_KEY` | recommended | without it the app still works as plain semantic search |
| `LOG_FORMAT` | no | forced to `json` by the prod compose file |

Secrets live only in `.env` (git-ignored, `chmod 600`). For a managed platform, set the same
names as environment variables or secrets instead.

## 3. Data and index (one-off, or on data updates)

Download the dataset (collected from MyAnimeList by the author and published on Kaggle:
[`furkanark/myanimelist-top-10000-anime-dataset`](https://www.kaggle.com/datasets/furkanark/myanimelist-top-10000-anime-dataset))
into `data/raw/`, either from the Kaggle page or with the Kaggle CLI:

```bash
kaggle datasets download furkanark/myanimelist-top-10000-anime-dataset -p data/raw --unzip
```

Then:

```bash
make ingest               # validate + clean  -> data/processed/anime.parquet
make enrich               # optional: English titles / full genres via Jikan (~3 h, resumable)
make ingest               # re-run if you enriched
make embed GPU=1          # ~1 min on a GPU; drop GPU=1 on CPU-only machines (~15 min)
make index                # sync to Qdrant (idempotent; --recreate after a model change)
```

Every stage is idempotent: re-running it produces the same result, and the embedding cache
means an interrupted run resumes instead of starting over. You can build the index on a
workstation with a GPU and serve from a small CPU VM, since both talk to the same Qdrant cluster.

## 4. Run

```bash
make prod     # docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

- Only `web` is published (port 80). The API is reachable only through it at `/api`.
- The API becomes healthy after about 20 s (model load and catalog scan). `web` waits for it.
- Logs are JSON with rotation (`docker compose logs -f api`).

Put TLS in front. The simplest option is Caddy on the host:

```
anime.example.com {
    reverse_proxy localhost:80
}
```

## 5. Operations

| Task | Command |
|---|---|
| Health | `curl localhost/api/health` (liveness), `/api/ready` (Qdrant and models) |
| Update code | `git pull && make prod` |
| Re-scraped data | `make ingest embed index`: unchanged anime are cache hits; removed anime are deleted from Qdrant |
| Switch embedding model | set `LOCAL_EMBEDDING_MODEL`, `make embed GPU=1`, `make index ARGS=--recreate` |
| Evaluate a change | `make eval GPU=1 ARGS="--models local:A,local:B --templates synopsis_only"` |

## Hardening checklist (not needed for a demo)

- Rate-limit `/api/chat` and `/api/search` at the proxy: each explained request costs a Gemini call.
- Run more API replicas behind the proxy. They are stateless apart from the shared query-embedding cache, which is best-effort.
- Turn off `/docs` in public deployments if you don't want the schema exposed.
