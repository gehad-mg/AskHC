"""
Document management API routes.
Handles document upload, indexing, and management.
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from ...config import settings
from ...models.schemas import DocumentUploadResponse
from ...services.document_loader import document_loader_service
from ...core.vectorstore import vectorstore_manager

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Upload and index a document.

    - **file**: PDF or TXT file to upload

    The document will be processed, chunked, and added to the vector store.
    """
    # Validate file extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    supported = document_loader_service.get_supported_extensions()

    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {supported}",
        )

    # Save file to documents directory
    save_path = settings.DOCUMENTS_DIR / filename

    # Ensure directory exists
    settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and index the document
        chunks_created = document_loader_service.load_and_index(str(save_path))

        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully",
            filename=filename,
            chunks_created=chunks_created,
            total_documents=vectorstore_manager.get_document_count(),
        )

    except Exception as e:
        # Clean up on error
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
) -> dict:
    """
    Upload and index multiple documents.

    - **files**: List of PDF or TXT files to upload

    All documents will be processed and added to the vector store.
    """
    results = []
    total_chunks = 0

    for file in files:
        filename = file.filename or "unknown"
        ext = Path(filename).suffix.lower()
        supported = document_loader_service.get_supported_extensions()

        if ext not in supported:
            results.append({
                "filename": filename,
                "status": "error",
                "message": f"Unsupported file type: {ext}",
            })
            continue

        save_path = settings.DOCUMENTS_DIR / filename
        settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            chunks = document_loader_service.load_and_index(str(save_path))
            total_chunks += chunks

            results.append({
                "filename": filename,
                "status": "success",
                "chunks_created": chunks,
            })

        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            results.append({
                "filename": filename,
                "status": "error",
                "message": str(e),
            })

    return {
        "results": results,
        "total_chunks_created": total_chunks,
        "total_documents_in_store": vectorstore_manager.get_document_count(),
    }


@router.get("/list")
async def list_documents() -> dict:
    """
    List all uploaded documents.

    Returns list of files in the documents directory.
    """
    docs_dir = settings.DOCUMENTS_DIR

    if not docs_dir.exists():
        return {"documents": [], "count": 0}

    documents = []
    for ext in document_loader_service.get_supported_extensions():
        for file_path in docs_dir.glob(f"*{ext}"):
            documents.append({
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "extension": file_path.suffix,
            })

    return {
        "documents": documents,
        "count": len(documents),
        "vectors_in_store": vectorstore_manager.get_document_count(),
    }


@router.delete("/clear")
async def clear_documents() -> dict:
    """
    Clear all documents from the vector store.

    WARNING: This is destructive and cannot be undone.
    Files are kept in the documents directory.
    """
    vectorstore_manager.clear()

    return {
        "message": "Vector store cleared",
        "vectors_remaining": vectorstore_manager.get_document_count(),
    }


@router.post("/reindex")
async def reindex_documents() -> dict:
    """
    Re-index all documents in the documents directory.

    Clears the vector store and re-processes all files.
    """
    # Clear existing vectors
    vectorstore_manager.clear()

    # Re-index all documents
    total_chunks = document_loader_service.load_directory()

    return {
        "message": "Documents re-indexed successfully",
        "total_chunks": total_chunks,
        "total_documents": vectorstore_manager.get_document_count(),
    }
