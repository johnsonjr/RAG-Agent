# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the interactive CLI:**
```bash
python scripts/run.py
```

**Run tests:**
```bash
pytest
```

**Lint:**
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Max line length is 127 characters.

## Architecture

This is a minimal, stdlib-only Python RAG system. No external packages are required to run the core logic.

**Pipeline flow:** `Query → SimpleRetriever (TF-IDF) → EchoLLM → Response`

**Core modules** (`src/rag_agent/`):
- `retriever.py` — `SimpleRetriever`: tokenizes with regex, builds TF-IDF vectors, ranks documents by cosine similarity. Returns `(document, score)` tuples.
- `llm.py` — `EchoLLM`: placeholder LLM that echoes prompts. Designed to be replaced with a real API (OpenAI, Anthropic, etc.). Uses `OPENAI_API_KEY` from environment when integrated.
- `pipeline.py` — `RAGPipeline`: wires retriever + LLM together. `query()` returns the LLM response; `query_with_scores()` also returns retrieval scores.

**Entry point** (`scripts/run.py`): loads corpus from `data/sample.txt` (documents split by double newlines), runs an interactive query loop.

**Tests** (`tests/test_retriever.py`): unit tests for retrieval correctness. Run via `pytest`.

## Extending the System

To replace `EchoLLM` with a real LLM, implement a class with `complete(prompt: str) -> str` and `complete_with_context(query: str, context: List[str]) -> str` methods, then pass it to `RAGPipeline`. For semantic search, replace `SimpleRetriever` with a vector-embedding-based retriever exposing the same `query(q, k)` interface.
