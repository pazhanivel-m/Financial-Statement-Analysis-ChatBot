"""
Central configuration — all settings are pulled from environment variables.
Update the .env file (copied from .env.example) before running the app.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure OpenAI — Chat (GPT-4o) ─────────────────────────────────────────────
    azure_openai_chat_api_key: str
    azure_openai_chat_endpoint: str             # e.g. https://<resource>.openai.azure.com/
    azure_openai_chat_deployment: str           # e.g. "gpt-4o"
    azure_openai_api_version: str = "2024-08-01-preview"

    # ── Azure OpenAI — Embeddings (text-embedding-3-large) ────────────────────────
    azure_openai_embedding_api_key: str
    azure_openai_embedding_endpoint: str        # e.g. https://<resource>.openai.azure.com/
    azure_openai_embedding_deployment: str      # e.g. "text-embedding-3-large"

    # text-embedding-3-large supports up to 3072; reduce if storage is a concern
    embedding_dimensions: int = 1536

    # ── PostgreSQL / pgvector ─────────────────────────────────────────────────
    postgres_dsn: str                           # e.g. postgresql+asyncpg://user:pass@host:5432/db

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # How many chunks each retriever fetches before RRF fusion
    retrieval_top_k: int = 5
    # Final number of chunks passed to the LLM as context
    context_top_n: int = 5
    # RRF constant — 60 is the widely-cited empirical sweet spot
    rrf_k: int = 60

    # ── Document processing ───────────────────────────────────────────────────
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "FSA Chatbot"
    app_version: str = "1.0.0"
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere instead of instantiating directly."""
    return Settings()
