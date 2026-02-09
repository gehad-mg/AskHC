# AskHC - Document Question Answering System

A RAG (Retrieval Augmented Generation) based application for querying documents using natural language.

## Features

- Upload and index PDF/TXT documents
- Ask questions about your documents
- Get AI-generated answers with source citations
- Simple, clean chat interface
- Modular, scalable architecture

## Project Structure

```
AskHC/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── api/routes/          # API endpoints
│   │   ├── core/                # RAG pipeline components
│   │   │   ├── embeddings.py    # Embedding model
│   │   │   ├── vectorstore.py   # ChromaDB operations
│   │   │   ├── llm.py           # LLM setup
│   │   │   └── rag_chain.py     # RAG chain
│   │   ├── services/            # Business logic
│   │   └── models/              # Pydantic schemas
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/
│   ├── documents/               # Uploaded documents
│   └── chroma_db/               # Vector store
└── .env.example
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run the Backend

```bash
cd backend
uvicorn app.main:app --reload
```

### 4. Open the Frontend

Open `frontend/index.html` in your browser, or serve it:

```bash
cd frontend
python -m http.server 3000
```

Then visit: http://localhost:3000

## API Endpoints

### Chat

- `POST /api/chat/ask` - Ask a question
- `GET /api/chat/history` - Get chat history
- `DELETE /api/chat/history` - Clear history
- `GET /api/chat/stats` - Get service stats

### Documents

- `POST /api/documents/upload` - Upload a document
- `POST /api/documents/upload-multiple` - Upload multiple documents
- `GET /api/documents/list` - List uploaded documents
- `POST /api/documents/reindex` - Re-index all documents
- `DELETE /api/documents/clear` - Clear vector store

### Health

- `GET /health` - Health check

## Configuration

Edit `backend/app/config.py` or use environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| EMBEDDING_MODEL_NAME | all-MiniLM-L6-v2 | Embedding model |
| LLM_MODEL_NAME | mistralai/Mistral-7B-Instruct-v0.3 | LLM model |
| CHUNK_SIZE | 500 | Document chunk size |
| RETRIEVER_K | 5 | Documents to retrieve |

## Extending the System

### Change Embedding Model

Edit `config.py`:
```python
EMBEDDING_MODEL_NAME = "your-preferred-model"
```

### Change LLM

1. Update `config.py`:
```python
LLM_MODEL_NAME = "your-model"
```

2. Update the prompt template in `rag_chain.py` if needed.

### Add Document Types

Edit `services/document_loader.py`:
1. Add to `SUPPORTED_EXTENSIONS`
2. Create a `_load_xxx` method
3. Update `load_file` to handle the type

## License

MIT
