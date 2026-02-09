"""
Pydantic schemas for API request and response models.
These define the structure of data exchanged with the API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask the RAG system",
    )
    include_sources: bool = Field(
        default=False,
        description="Whether to include source documents in response",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the company policy on remote work?",
                "include_sources": True,
            }
        }


class SourceDocument(BaseModel):
    """Model for source document information."""

    content: str = Field(description="Preview of document content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata (source, page, etc.)",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(description="The generated answer")
    status: str = Field(description="Response status (success, error, etc.)")
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp of response",
    )
    sources: Optional[List[SourceDocument]] = Field(
        default=None,
        description="Source documents used for the answer",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "According to company policy, employees can work remotely up to 2 days per week.",
                "status": "success",
                "timestamp": "2024-01-15T10:30:00",
                "sources": [
                    {
                        "content": "Remote work policy excerpt...",
                        "metadata": {"source": "hr_policy.pdf", "page": 5},
                    }
                ],
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload endpoint."""

    message: str = Field(description="Status message")
    filename: str = Field(description="Name of uploaded file")
    chunks_created: int = Field(description="Number of chunks created")
    total_documents: int = Field(description="Total documents in store")


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""

    documents_indexed: int = Field(description="Number of document chunks indexed")
    conversation_length: int = Field(description="Number of messages in history")
    status: str = Field(description="Service status")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(description="Service health status")
    app_name: str = Field(description="Application name")
    version: str = Field(description="Application version")
