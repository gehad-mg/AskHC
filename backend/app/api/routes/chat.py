"""
Chat API routes.
Handles chat interactions with the RAG system.
"""

from fastapi import APIRouter, HTTPException

from ...models.schemas import ChatRequest, ChatResponse, StatsResponse
from ...services.chat_service import chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest) -> ChatResponse:
    """
    Ask a question to the RAG system.

    - **question**: The question to ask (required)
    - **include_sources**: Whether to include source documents (default: True)

    Returns the generated answer based on indexed documents.
    """
    result = chat_service.ask(
        question=request.question,
        include_sources=request.include_sources,
    )

    return ChatResponse(**result)


@router.get("/history")
async def get_chat_history():
    """
    Get the conversation history.

    Returns list of messages in the current conversation.
    """
    return {
        "history": chat_service.get_history(),
        "count": len(chat_service.get_history()),
    }


@router.delete("/history")
async def clear_chat_history():
    """
    Clear the conversation history.

    Resets the conversation to a fresh state.
    """
    chat_service.clear_history()
    return {"message": "Conversation history cleared"}


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get chat service statistics.

    Returns information about indexed documents and conversation state.
    """
    stats = chat_service.get_stats()
    return StatsResponse(**stats)
