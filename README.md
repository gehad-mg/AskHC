# AskHC - Document Question Answering System

A RAG (Retrieval Augmented Generation) based application for querying documents using natural language. Uses Nebius API for embeddings and LLM.

## Features

- Index PDF/TXT documents
- Ask questions about your documents in English or Arabic
- Multilingual support with bge-multilingual-gemma2 embeddings
- Conversation memory (follow-up questions work)
- Simple, clean chat interface

## Project Structure

```
AskHC/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── api/routes/          # API endpoints
│   │   ├── core/                # RAG pipeline components
│   │   │   ├── embeddings.py    # Nebius API embeddings
│   │   │   ├── vectorstore.py   # ChromaDB operations
│   │   │   ├── llm.py           # Nebius API LLM
│   │   │   └── rag_chain.py     # RAG chain
│   │   ├── services/            # Business logic
│   │   └── models/              # Pydantic schemas
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/
│   ├── documents/               # Place your PDF/TXT files here
│   └── chroma_db/               # Vector store (auto-generated)
├── load_documents.py            # Script to index documents
├── start.bat                    # Windows startup script
└── .env                         # Configuration file
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Edit the `.env` file in the project root with your Nebius API key:

```
NEBIUS_API_KEY=your_nebius_api_key_here
```

### 3. Add Documents

Place your PDF or TXT files in the `data/documents/` folder.

### 4. Index Documents

```bash
python load_documents.py
```

This will process all documents and create embeddings in the vector store.

### 5. Run the Application

**Option A: Using start.bat (Windows)**
```bash
start.bat
```

**Option B: Manual**
```bash
cd backend
uvicorn app.main:app --reload
```

Then open your browser to: http://localhost:8000

## Configuration

Edit `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| NEBIUS_API_KEY | (required) | Your Nebius API key |
| NEBIUS_BASE_URL | https://api.tokenfactory.nebius.com/v1/ | Nebius API endpoint |
| EMBEDDING_MODEL_NAME | BAAI/bge-multilingual-gemma2 | Embedding model (multilingual) |
| LLM_MODEL_NAME | meta-llama/Meta-Llama-3.1-8B-Instruct-fast | LLM model |
| LLM_MAX_TOKENS | 1024 | Max response tokens |
| LLM_TEMPERATURE | 0.7 | Response creativity (0-1) |
| CHUNK_SIZE | 500 | Document chunk size |
| CHUNK_OVERLAP | 50 | Overlap between chunks |
| RETRIEVER_K | 5 | Number of documents to retrieve |

## Re-indexing Documents

If you add new documents or want to rebuild the index:

```bash
python load_documents.py
```

This clears the existing index and re-processes all documents.

## API Endpoints

### Chat

- `POST /api/chat/ask` - Ask a question
- `GET /api/chat/history` - Get chat history
- `DELETE /api/chat/history` - Clear history
- `GET /api/chat/stats` - Get service stats

### Documents

- `POST /api/documents/upload` - Upload a document
- `GET /api/documents/list` - List uploaded documents
- `POST /api/documents/reindex` - Re-index all documents
- `DELETE /api/documents/clear` - Clear vector store

### Health

- `GET /health` - Health check

## Customization

### Change the System Prompt

Edit `backend/app/core/rag_chain.py`:
```python
SYSTEM_PROMPT = """Your custom prompt here"""
```

### Change Models

Edit `.env`:
```
EMBEDDING_MODEL_NAME=your-embedding-model
LLM_MODEL_NAME=your-llm-model
```

## License

MIT
