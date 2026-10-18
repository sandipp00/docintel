"""Central configuration, loaded once from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime settings. Secrets use SecretStr so they never print in logs/reprs."""

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # LLM
    llm_provider: Literal["groq", "gemini", "none"] = "groq"
    groq_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-flash-lite-latest"
    llm_timeout_seconds: float = 30.0
    llm_rpm: int = 14  # Gemini free tier allows 15 requests/minute; stay just under

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    tesseract_cmd: str | None = None

    # Limits & retention
    max_upload_mb: int = Field(default=10, ge=1, le=50)
    retention_minutes: int = Field(default=60, ge=1)
    rate_limit_default: str = "60/minute"
    rate_limit_upload: str = "10/minute"
    rate_limit_ask: str = "20/minute"
    allowed_extensions: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".docx")
    max_pages: int = 30
    max_image_pixels: int = 40_000_000
    ocr_min_chars_per_page: int = 30  # digital PDF pages with less text than this are OCR'd
    ocr_dpi: int = 200

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    min_retrieval_score: float = 0.12 # cosine; below this the question is treated as off-topic
    min_support: float = 0.5  # lexical support an LLM answer needs from its cited sources

    # Server
    cors_origins: str = "*"
    log_level: str = "INFO"

    # Paths
    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    web_dir: Path = ROOT_DIR / "web"
    evaluation_dir: Path = ROOT_DIR / "evaluation"
    sample_dir: Path = ROOT_DIR / "sample_docs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        """True when a provider is selected and its key is present."""
        key = {"groq": self.groq_api_key, "gemini": self.gemini_api_key}.get(self.llm_provider)
        return key is not None and bool(key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
