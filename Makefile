# Everything runs in Docker; nothing is installed on the host.
COMPOSE := docker compose
DEV     := $(COMPOSE) run --rm dev
APP     := $(COMPOSE) run --rm --build api

.DEFAULT_GOAL := help
.PHONY: help env build lock up down logs ingest embed index eval pipeline test lint format typecheck check shell

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from the template (does not overwrite)
	@test -f .env || (cp .env.example .env && echo "created .env — fill in GEMINI_API_KEY and QDRANT_*")

build: ## Build runtime and dev images
	$(COMPOSE) --profile dev build

lock: ## (Re)generate uv.lock inside the dev container
	$(DEV) uv lock

up: ## Start API + UI (uses the Qdrant cluster from .env)
	$(COMPOSE) up -d --build

down: ## Stop all services
	$(COMPOSE) --profile local-qdrant down

logs: ## Tail service logs
	$(COMPOSE) logs -f

# --- pipeline stages (each idempotent, runnable independently) ---
ingest: ## Raw CSV -> data/processed/anime.parquet
	$(APP) anime-rec ingest
embed: ## Parquet -> cached Gemini embeddings (ARGS="--limit 20" for a trial)
	$(APP) anime-rec embed $(ARGS)
index: ## Embeddings -> Qdrant (ARGS="--recreate" to rebuild)
	$(APP) anime-rec index $(ARGS)
eval: ## Retrieval evaluation report
	$(APP) anime-rec eval
pipeline: ingest embed index ## Run all pipeline stages

# --- quality ---
test: ## Run pytest
	$(DEV) pytest
lint: ## Ruff lint + format check
	$(DEV) ruff check .
	$(DEV) ruff format --check .
format: ## Auto-format, then auto-fix lint
	$(DEV) ruff format .
	$(DEV) ruff check --fix .
typecheck: ## mypy
	$(DEV) mypy src
check: lint typecheck test ## Everything CI runs

shell: ## Shell inside the dev container
	$(DEV) bash
