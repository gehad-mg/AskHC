"""
AskHC - Main FastAPI Application

This is the entry point for the AskHC backend API.
Run with: uvicorn app.main:app --reload

The API provides:
- / - Chat interface (frontend)
- /api/chat - Chat endpoints for RAG queries
- /api/documents - Document upload and management
- /health - Health check endpoint
- /docs - API documentation
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .config import settings
from .api.routes import chat_router, documents_router


# Path to frontend files
FRONTEND_PATH = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs on startup and shutdown.
    """
    # Startup
    print(f"[{settings.APP_NAME}] Starting application...")
    print(f"[{settings.APP_NAME}] Documents directory: {settings.DOCUMENTS_DIR}")
    print(f"[{settings.APP_NAME}] ChromaDB directory: {settings.CHROMA_DB_DIR}")
    print(f"[{settings.APP_NAME}] Frontend: http://localhost:{settings.API_PORT}")

    # Ensure directories exist
    settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    print(f"[{settings.APP_NAME}] Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="RAG-based Question Answering API for document queries",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the API.
    """
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# Serve frontend at root
@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """
    Serve the chat interface.
    """
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found", "api_docs": "/docs"}


# Mount static files for frontend assets (CSS, JS)
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
