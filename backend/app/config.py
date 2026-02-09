"""
Configuration settings for the AskHC RAG application.
All settings are centralized here for easy modification.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Defaults are provided for development.
    """

    # ===========================================
    # Application Settings
    # ===========================================
    APP_NAME: str = "AskHC"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ===========================================
    # Path Settings
    # ===========================================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"

    # ===========================================
    # Nebius API Settings
    # ===========================================
    NEBIUS_API_KEY: str = ""
    NEBIUS_BASE_URL: str = "https://api.tokenfactory.nebius.com/v1/"

    # ===========================================
    # Embedding Model Settings (Nebius API)
    # ===========================================
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-multilingual-gemma2"

    # ===========================================
    # Vector Store Settings
    # ===========================================
    CHROMA_COLLECTION_NAME: str = "askhc_documents"
    RETRIEVER_K: int = 5  # Number of documents to retrieve

    # ===========================================
    # Document Processing Settings
    # ===========================================
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ===========================================
    # LLM Settings (Nebius API)
    # ===========================================
    LLM_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast"
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7

    # ===========================================
    # API Settings
    # ===========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from old .env files


# Global settings instance
settings = Settings()
