"""
Chat service for handling user conversations.
Provides the main interface for the chat API.

This service:
1. Processes user messages
2. Queries the RAG chain
3. Returns formatted responses
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.rag_chain import rag_chain
from ..core.vectorstore import vectorstore_manager


class ChatService:
    """
    Service for handling chat interactions.
    Provides a clean interface between API and RAG chain.
    """

    def __init__(self):
        """Initialize chat service."""
        self._conversation_history: List[Dict[str, str]] = []

    def ask(
        self,
        question: str,
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a user question and return an answer.

        Args:
            question: The user's question.
            include_sources: Whether to include source documents.

        Returns:
            Dictionary with answer and metadata.
        """
        # Validate question
        question = question.strip()
        if not question:
            return {
                "answer": "Please provide a question.",
                "status": "error",
            }

        # Check if documents are indexed
        doc_count = vectorstore_manager.get_document_count()
        if doc_count == 0:
            return {
                "answer": "No documents have been indexed yet. "
                         "Please upload documents first.",
                "status": "no_documents",
            }

        # Query RAG chain with conversation history for context
        try:
            result = rag_chain.query(
                question=question,
                return_sources=include_sources,
                conversation_history=self._conversation_history,
            )

            # Add metadata
            response = {
                "answer": result["answer"],
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

            if include_sources and "sources" in result:
                response["sources"] = result["sources"]

            # Store in conversation history
            self._conversation_history.append({
                "role": "user",
                "content": question,
            })
            self._conversation_history.append({
                "role": "assistant",
                "content": result["answer"],
            })

            return response

        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "status": "error",
            }

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.

        Returns:
            List of message dictionaries with role and content.
        """
        return self._conversation_history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history = []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with stats about the service.
        """
        return {
            "documents_indexed": vectorstore_manager.get_document_count(),
            "conversation_length": len(self._conversation_history),
            "status": "ready" if vectorstore_manager.get_document_count() > 0 else "waiting_for_documents",
        }


# Global service instance
chat_service = ChatService()
