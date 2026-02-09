"""
Vector store manager for the RAG pipeline.
Handles ChromaDB operations for document storage and retrieval.

To swap vector stores:
1. Replace Chroma with your preferred vector store
2. Update the _create_store and _load_store methods
"""

from typing import List, Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from ..config import settings
from .embeddings import embedding_manager


class VectorStoreManager:
    """
    Manages the ChromaDB vector store.
    Handles creating, loading, and querying the store.
    """

    _instance: Optional["VectorStoreManager"] = None
    _vectorstore: Optional[Chroma] = None

    def __new__(cls) -> "VectorStoreManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize vector store manager."""
        if self._vectorstore is None:
            self._load_or_create_store()

    def _load_or_create_store(self) -> None:
        """
        Load existing vector store or create a new one.
        Persists to disk at CHROMA_DB_DIR.
        """
        persist_dir = str(settings.CHROMA_DB_DIR)
        collection_name = settings.CHROMA_COLLECTION_NAME

        # Check if store exists
        if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            print(f"[VectorStore] Loading existing store from: {persist_dir}")
            self._vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_manager.model,
                collection_name=collection_name,
            )
            count = self._vectorstore._collection.count()
            print(f"[VectorStore] Loaded {count} vectors")
        else:
            print(f"[VectorStore] Creating new store at: {persist_dir}")
            self._vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_manager.model,
                collection_name=collection_name,
            )
            print("[VectorStore] Empty store created")

    @property
    def store(self) -> Chroma:
        """
        Get the vector store instance.

        Returns:
            Chroma: The ChromaDB vector store.
        """
        if self._vectorstore is None:
            self._load_or_create_store()
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects to add.
        """
        if not documents:
            print("[VectorStore] No documents to add")
            return

        print(f"[VectorStore] Adding {len(documents)} documents...")
        self.store.add_documents(documents)
        count = self.store._collection.count()
        print(f"[VectorStore] Total vectors: {count}")

    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: The search query.
            k: Number of results to return (default from config).

        Returns:
            List of matching Document objects.
        """
        k = k or settings.RETRIEVER_K
        return self.store.similarity_search(query, k=k)

    def get_retriever(self, k: Optional[int] = None) -> VectorStoreRetriever:
        """
        Get a retriever for the RAG chain.

        Args:
            k: Number of documents to retrieve (default from config).

        Returns:
            VectorStoreRetriever configured for the store.
        """
        k = k or settings.RETRIEVER_K
        return self.store.as_retriever(search_kwargs={"k": k})

    def get_document_count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            Number of documents/vectors stored.
        """
        return self.store._collection.count()

    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        Use with caution - this is destructive.
        """
        print("[VectorStore] Clearing all documents...")

        # Delete the collection using ChromaDB's API (doesn't require file deletion)
        if self._vectorstore is not None:
            try:
                # Get the underlying client and delete the collection
                client = self._vectorstore._client
                client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
                print("[VectorStore] Collection deleted")
            except Exception as e:
                print(f"[VectorStore] Warning during delete: {e}")

        self._vectorstore = None

        # Recreate empty store with new collection
        self._load_or_create_store()
        print("[VectorStore] Store cleared")


# Global instance for easy access
vectorstore_manager = VectorStoreManager()
