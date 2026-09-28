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

    # --- Gemini ---
    gemini_api_key: SecretStr | None = None
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_chat_model: str = "gemini-2.5-flash"
    embedding_dim: int = 768  # Matryoshka sizes supported by gemini-embedding-001
    embedding_batch_size: int = Field(default=50, ge=1, le=100)
    # Client-side throttle in *texts* per minute: the quota counts each text in a batch.
    # Free tier allows 100/min; 90 leaves headroom for query embeddings from the API.
    embedding_texts_per_minute: int = Field(default=90, ge=1)
    embedding_max_retries: int = Field(default=6, ge=0)
    document_template: str = "full"  # see embeddings/documents.py TEMPLATES

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "anime"

    # --- Data paths ---
    data_dir: Path = Path("data")

    # --- Serving ---
    api_url: str = "http://api:8000"  # used by the Streamlit UI
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
