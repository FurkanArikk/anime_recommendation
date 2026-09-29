# syntax=docker/dockerfile:1.7
# One image serves the API, the Streamlit UI and the pipeline CLI (different commands).
# Targets:  runtime (default, no dev deps)  |  dev (adds pytest/ruff/mypy, used by `make test`)
# Build arg TORCH_VARIANT=cpu|gpu selects the PyTorch build (GPU = CUDA 13.0 wheels).

FROM python:3.14-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # Hugging Face model downloads land in the bind-mounted data dir: downloaded once,
    # reused across containers and rebuilds, never baked into the image.
    HF_HOME=/app/data/hf-cache
WORKDIR /app
ARG TORCH_VARIANT=cpu
# GPU only: Triton JIT-compiles a small C driver shim at first use (torch routes some ops,
# e.g. in Gemma3 attention, to Triton kernels), so it needs a C compiler at runtime.
RUN if [ "$TORCH_VARIANT" = "gpu" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# ---- runtime -----------------------------------------------------------------
FROM base AS runtime
# Dependencies first (cached layer), then project source.
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --extra ${TORCH_VARIANT}
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra ${TORCH_VARIANT}
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app /app
USER app
EXPOSE 8000 8501
CMD ["anime-rec", "serve"]

# ---- dev ---------------------------------------------------------------------
FROM base AS dev
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --extra ${TORCH_VARIANT}
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra ${TORCH_VARIANT}
CMD ["pytest"]
