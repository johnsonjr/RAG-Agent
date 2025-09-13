# RAG-Agent

A minimal Python implementation of a Retrieval-Augmented Generation (RAG) agent with TF-IDF retrieval.

## Features

- Simple TF-IDF based document retriever with cosine similarity
- Modular pipeline architecture (retriever → LLM → response)
- Interactive CLI for testing queries
- No external dependencies (uses only Python stdlib)

## Getting Started

### Run the CLI

```bash
python scripts/run.py
```

This will load documents from `data/sample.txt` and let you query them interactively.

### Run Tests

```bash
python tests/test_retriever.py
```

## Project Structure

- `src/rag_agent/` - Core RAG implementation
  - `retriever.py` - TF-IDF document retrieval
  - `llm.py` - LLM interface (currently echo mode)
  - `pipeline.py` - RAG pipeline orchestration
- `scripts/run.py` - Interactive CLI
- `data/sample.txt` - Sample corpus
- `tests/` - Unit tests

## Next Steps

- Replace `EchoLLM` with actual LLM integration (OpenAI, Anthropic, etc.)
- Add vector embeddings for semantic search
- Implement document chunking for larger texts
- Add evaluation metrics
