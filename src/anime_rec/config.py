"""Application settings, loaded from environment variables (and a .env file if present).

Every tunable lives here so that no module reads os.environ directly and no secret is
hard-coded. Settings are cached: call `get_settings()` rather than instantiating `Settings`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Embeddings ---
    embedding_provider: Literal["local", "gemini"] = "local"
    document_template: str = "synopsis_only"  # beat "full" for every model in eval

    # Local (Hugging Face sentence-transformers); dimension comes from the model.
    local_embedding_model: str = "google/embeddinggemma-300m"  # best in eval (see README)
    embedding_device: str = "auto"  # auto | cpu | cuda
    local_batch_size: int = Field(default=64, ge=1)

    # --- Gemini (chat/explanations; optional embedding provider) ---
    gemini_api_key: SecretStr | None = None
    gemini_chat_model: str = "gemini-2.5-flash"
    # Tried in order when the primary is overloaded (503) or rate-limited (429).
    gemini_chat_fallback_models: list[str] = ["gemini-3.5-flash-lite"]
    # Gemini 2.5 "thinking" tripled latency (13.7s -> 3s at 0) with no visible quality gain for
    # this short, grounded task. None = model default. Gemini 3 uses levels, not budgets.
    gemini_thinking_budget: int | None = 0
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768  # Gemini output size (Matryoshka: 768 | 1536 | 3072)
    embedding_batch_size: int = Field(default=50, ge=1, le=100)
    # Client-side throttle in *texts* per minute: the quota counts each text in a batch.
    # Free tier allows 100/min; 90 leaves headroom for query embeddings from the API.
    embedding_texts_per_minute: int = Field(default=90, ge=1)
    embedding_max_retries: int = Field(default=6, ge=0)

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "anime"

    # --- Data paths ---
    data_dir: Path = Path("data")

    # --- Serving ---
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator("embedding_dim")
    @classmethod
    def _supported_dim(cls, v: int) -> int:
        if v not in (768, 1536, 3072):
            raise ValueError("embedding_dim must be 768, 1536 or 3072")
        return v

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def jikan_dir(self) -> Path:
        return self.raw_dir / "jikan"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def processed_parquet(self) -> Path:
        return self.processed_dir / "anime.parquet"

    @property
    def embedding_cache_path(self) -> Path:
        return self.cache_dir / "embeddings.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
