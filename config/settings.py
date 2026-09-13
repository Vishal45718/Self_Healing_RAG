"""Centralized configuration module for Self-Healing RAG.

Loads configuration from environment variables and an optional .env file.
Provides sensible defaults without hardcoding any secrets or credentials.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and runtime parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Selection ("gemini" or "huggingface")
    llm_provider: str = "gemini"

    # Gemini Settings
    gemini_api_key: Optional[str] = None
    gemini_model_id: str = "gemini-3.6-flash"

    # Hugging Face Settings
    hf_token: Optional[str] = None
    hf_provider: str = "together"
    llm_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chroma Vector Database Settings
    chroma_path: str = "./data/chroma"

    # Self-Healing Pipeline Limits
    max_retries: int = 3

    # Document Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @property
    def chroma_directory(self) -> Path:
        """Return the resolved Chroma persistence path."""
        return Path(self.chroma_path).resolve()

    @property
    def active_model_id(self) -> str:
        """Return the active model ID based on the configured provider."""
        if self.llm_provider.lower() == "gemini":
            return self.gemini_model_id
        return self.llm_model_id


# Centralized settings instance
settings = Settings()
