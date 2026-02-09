"""
Embedding model manager for the RAG pipeline.
Uses Nebius API for multilingual embeddings (bge-multilingual-gemma2).

To swap embedding models:
1. Change EMBEDDING_MODEL_NAME in config.py
2. Ensure your NEBIUS_API_KEY is set in .env
"""

from typing import List, Optional
from openai import OpenAI
from langchain_core.embeddings import Embeddings

from ..config import settings


class NebiusEmbeddings(Embeddings):
    """
    Custom embeddings class using Nebius API.
    Compatible with LangChain's Embeddings interface.
    """

    def __init__(self):
        """Initialize Nebius client."""
        if not settings.NEBIUS_API_KEY:
            raise ValueError(
                "NEBIUS_API_KEY not set. Please add it to your .env file."
            )

        self.client = OpenAI(
            base_url=settings.NEBIUS_BASE_URL,
            api_key=settings.NEBIUS_API_KEY,
        )
        self.model = settings.EMBEDDING_MODEL_NAME

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        # Process in batches to avoid API limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            for item in response.data:
                embeddings.append(item.embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding


class EmbeddingManager:
    """
    Manages the embedding model lifecycle.
    Singleton pattern ensures only one client instance.
    """

    _instance: Optional["EmbeddingManager"] = None
    _embeddings: Optional[NebiusEmbeddings] = None

    def __new__(cls) -> "EmbeddingManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize embedding manager (only runs once due to singleton)."""
        if self._embeddings is None:
            self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model via Nebius API."""
        print(f"[Embeddings] Connecting to Nebius API...")
        print(f"[Embeddings] Model: {settings.EMBEDDING_MODEL_NAME}")

        self._embeddings = NebiusEmbeddings()

        # Test the connection
        test_embedding = self._embeddings.embed_query("test")
        print(f"[Embeddings] Connected. Dimension: {len(test_embedding)}")

    @property
    def model(self) -> NebiusEmbeddings:
        """
        Get the embedding model instance.

        Returns:
            NebiusEmbeddings: The embedding model.
        """
        if self._embeddings is None:
            self._load_model()
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self.model.embed_documents(texts)


# Global instance for easy access
embedding_manager = EmbeddingManager()
