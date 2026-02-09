"""
Document loading service for the RAG pipeline.
Handles loading, processing, and chunking documents.

Supports:
- PDF files (with OCR fallback for scanned documents)
- Text files

To add new document types:
1. Add a new loader method (_load_xxx)
2. Register the extension in SUPPORTED_EXTENSIONS
3. Update the load_file method
"""

import os
from pathlib import Path
from typing import List, Optional, Set

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config import settings
from ..core.vectorstore import vectorstore_manager


# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentLoaderService:
    """
    Service for loading and processing documents.
    Handles chunking and vector store ingestion.
    """

    def __init__(self):
        """Initialize document loader with configured splitter."""
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        )

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file and return documents.

        Args:
            file_path: Path to the file.

        Returns:
            List of Document objects.

        Raises:
            ValueError: If file type is not supported.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        print(f"[DocumentLoader] Loading: {path.name}")

        if ext == ".pdf":
            docs = self._load_pdf(file_path)
        elif ext == ".txt":
            docs = self._load_text(file_path)
        else:
            docs = []

        print(f"[DocumentLoader] Loaded {len(docs)} pages from {path.name}")
        return docs

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file.

        Args:
            file_path: Path to the PDF.

        Returns:
            List of Document objects.
        """
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Check for empty pages (scanned PDFs)
        empty_pages = [
            doc for doc in docs if len(doc.page_content.strip()) < 10
        ]

        if empty_pages:
            print(
                f"[DocumentLoader] Found {len(empty_pages)} empty pages, "
                "attempting OCR..."
            )
            docs = self._ocr_pdf(file_path, docs)

        return docs

    def _ocr_pdf(
        self, file_path: str, original_docs: List[Document]
    ) -> List[Document]:
        """
        Apply OCR to scanned PDF pages.

        Args:
            file_path: Path to the PDF.
            original_docs: Original documents (some may be empty).

        Returns:
            Documents with OCR-extracted text.
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract

            # Keep non-empty pages
            result = [
                doc
                for doc in original_docs
                if len(doc.page_content.strip()) >= 10
            ]

            # OCR empty pages
            images = convert_from_path(file_path)
            for i, image in enumerate(images):
                # Check if this page was empty
                if i < len(original_docs):
                    if len(original_docs[i].page_content.strip()) < 10:
                        text = pytesseract.image_to_string(image)
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "page": i,
                                "ocr": True,
                            },
                        )
                        result.append(doc)
                        print(f"[DocumentLoader] OCR page {i + 1}: {len(text)} chars")

            return result

        except ImportError:
            print(
                "[DocumentLoader] OCR libraries not installed. "
                "Install: pip install pdf2image pytesseract"
            )
            return original_docs

    def _load_text(self, file_path: str) -> List[Document]:
        """
        Load a text file.

        Args:
            file_path: Path to the text file.

        Returns:
            List of Document objects.
        """
        loader = TextLoader(file_path)
        return loader.load()

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better retrieval.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of chunked documents.
        """
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)
        print(f"[DocumentLoader] Split into {len(chunks)} chunks")
        return chunks

    def load_and_index(self, file_path: str) -> int:
        """
        Load a file and add it to the vector store.

        Args:
            file_path: Path to the file.

        Returns:
            Number of chunks added.
        """
        # Load
        docs = self.load_file(file_path)

        # Chunk
        chunks = self.chunk_documents(docs)

        # Index
        if chunks:
            vectorstore_manager.add_documents(chunks)

        return len(chunks)

    def load_directory(self, directory_path: Optional[str] = None) -> int:
        """
        Load all supported files from a directory.

        Args:
            directory_path: Path to directory (default: config DOCUMENTS_DIR).

        Returns:
            Total number of chunks added.
        """
        dir_path = Path(directory_path or settings.DOCUMENTS_DIR)

        if not dir_path.exists():
            print(f"[DocumentLoader] Directory not found: {dir_path}")
            return 0

        total_chunks = 0
        files_processed = 0

        for ext in SUPPORTED_EXTENSIONS:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    chunks = self.load_and_index(str(file_path))
                    total_chunks += chunks
                    files_processed += 1
                except Exception as e:
                    print(f"[DocumentLoader] Error loading {file_path}: {e}")

        print(
            f"[DocumentLoader] Processed {files_processed} files, "
            f"{total_chunks} chunks total"
        )
        return total_chunks

    @staticmethod
    def get_supported_extensions() -> Set[str]:
        """Get the set of supported file extensions."""
        return SUPPORTED_EXTENSIONS


# Global service instance
document_loader_service = DocumentLoaderService()
