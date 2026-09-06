# Components

This directory contains the refactored components of the Hybrid RAG chatbot application, separated by concern:

## Files

- `chunking.py` - Document chunking functionality
- `embeddings.py` - Embedding model initialization and management
- `llm.py` - Language model initialization and management
- `retriever.py` - Retriever creation (BM25, vector, and hybrid)
- `vector_store.py` - Vector store (ChromaDB) initialization and management

## Usage

Each component exposes functions that are imported and used in `main.py`:

```python
from components.chunking import chunk_documents
from components.embeddings import get_embeddings_model
from components.llm import get_llm
from components.vector_store import initialize_vector_store, get_vector_retriever
from components.retriever import create_bm25_retriever, create_hybrid_retriever
```