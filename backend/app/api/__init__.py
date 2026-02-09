"""
API module containing route definitions.
"""

from .routes import chat_router, documents_router

__all__ = [
    "chat_router",
    "documents_router",
]
