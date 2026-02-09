"""
Core module containing the RAG pipeline components.
Each component is modular and can be easily swapped or updated.
"""

from .embeddings import EmbeddingManager
from .vectorstore import VectorStoreManager
from .llm import LLMManager
from .rag_chain import RAGChain

__all__ = [
    "EmbeddingManager",
    "VectorStoreManager",
    "LLMManager",
    "RAGChain",
]
