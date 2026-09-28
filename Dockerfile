# syntax=docker/dockerfile:1.7
# One image serves the API, the Streamlit UI and the pipeline CLI (different commands).
# Targets:  runtime (default, no dev deps)  |  dev (adds pytest/ruff/mypy, used by `make test`)

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# ---- runtime -----------------------------------------------------------------
FROM base AS runtime
# Dependencies first (cached layer), then project source.
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app /app
USER app
EXPOSE 8000 8501
CMD ["anime-rec", "serve"]

# ---- dev ---------------------------------------------------------------------
FROM base AS dev
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync
CMD ["pytest"]
