"""
Load Documents Script
=====================
Loads PDFs into the vector database.
Clears existing data first to avoid duplicates.

Usage:
1. Put your PDF files in: data/documents/
2. Run: python load_documents.py
3. Then start the server: run_backend.bat
"""

import sys
sys.path.insert(0, 'backend')

from pathlib import Path

# Set up paths
DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 50)
    print("AskHC - Document Loader")
    print("=" * 50)

    # Check for documents
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    txt_files = list(DOCUMENTS_DIR.glob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        print(f"\nNo documents found in: {DOCUMENTS_DIR.absolute()}")
        print("\nPlease add your PDF or TXT files to this folder and run again.")
        return

    print(f"\nFound {len(all_files)} document(s):")
    for f in all_files:
        print(f"  - {f.name}")

    # Import after path setup
    from app.services.document_loader import document_loader_service
    from app.core.vectorstore import vectorstore_manager

    # Clear existing data first
    existing_count = vectorstore_manager.get_document_count()
    if existing_count > 0:
        print(f"\nClearing existing {existing_count} vectors...")
        vectorstore_manager.clear()

    print("\nLoading documents...")

    # Load all documents
    total_chunks = document_loader_service.load_directory(str(DOCUMENTS_DIR))

    print("\n" + "=" * 50)
    print(f"Done! Loaded {total_chunks} chunks into vector database.")
    print(f"Total vectors: {vectorstore_manager.get_document_count()}")
    print("=" * 50)
    print("\nYou can now run: .\\run_backend.bat")


if __name__ == "__main__":
    main()
