"""
RAG Chain implementation for the AskHC application.
Combines retrieval and generation into a single pipeline.

To customize the RAG behavior:
1. Modify the SYSTEM_PROMPT for different response styles
2. Adjust RETRIEVER_K in config for more/fewer context documents
3. Override the format_docs method for custom formatting
"""

from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from ..config import settings
from .vectorstore import vectorstore_manager
from .llm import get_llm_manager


# ===========================================
# Prompt Configuration
# ===========================================
SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the question based ONLY on the following context.
Be concise and accurate. You can respond in the same language as the question."""



class RAGChain:
    """
    RAG (Retrieval Augmented Generation) chain.
    Retrieves relevant documents and generates answers.
    """

    _instance: Optional["RAGChain"] = None

    def __new__(cls) -> "RAGChain":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize RAG chain components."""
        print("[RAGChain] Prompt template configured")

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def query(
        self,
        question: str,
        return_sources: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: The user's question.
            return_sources: Whether to include source documents.
            conversation_history: Previous messages for context.

        Returns:
            Dictionary with 'answer' and optionally 'sources'.
        """
        print(f"[RAGChain] Processing question: {question[:50]}...")

        # Retrieve relevant documents
        docs = vectorstore_manager.similarity_search(question)
        context = self.format_docs(docs)

        # Build conversation context if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Include last 6 messages (3 exchanges) for context
            recent_history = conversation_history[-6:]
            conversation_context = "Previous conversation:\n"
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            conversation_context += "\n"

        # Build the user prompt with context
        user_prompt = f"""Context from documents:
{context}

{conversation_context}Current question: {question}

Answer:"""

        # Generate answer using Nebius API
        llm = get_llm_manager().llm
        answer = llm.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        result = {"answer": answer.strip()}

        # Optionally include sources
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                }
                for doc in docs
            ]

        return result

    def is_ready(self) -> bool:
        """Check if the chain is ready to accept queries."""
        return vectorstore_manager.get_document_count() > 0


# Global instance for easy access
rag_chain = RAGChain()
