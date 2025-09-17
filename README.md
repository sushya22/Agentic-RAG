# Agentic RAG Challenge — Dynamic Knowledge Assistant

This repository is a self-contained Python implementation of an **Agentic RAG (Retrieval-Augmented Generation)** system,
built for a hackathon challenge: *Building a Dynamic Knowledge Assistant*.

## Highlights
- Minimal RAG pipeline (document store, retriever, generator interface).
- "Agentic" behavior: planner + executor loop that decides when to retrieve, answer, or call tools.
- Plug-and-play embeddings: uses `sentence-transformers` if installed, otherwise falls back to TF-IDF vectors.
- FastAPI demo app for interactive testing.
- Unit tests demonstrating key behaviors.
- Small sample dataset included.

## Repo structure
- `src/` - main codebase
  - `main.py` - FastAPI app (demo)
  - `rag_agent.py` - core agent and pipeline
  - `embeddings.py` - embedding interface with fallback
  - `tools.py` - simple agent tools (calculator, file search)
  - `utils.py` - helper functions
- `data/sample_docs/` - sample documents used for retrieval
- `tests/` - pytest unit tests
- `requirements.txt` - suggested packages
- `README.md` - this file

## Requirements
Python 3.9+ recommended.

Suggested install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you cannot install `sentence-transformers` or `faiss-cpu`, the code will run using a TF-IDF fallback retriever.

## Run demo (FastAPI)
```bash
cd src
uvicorn main:app --reload --port 8000
```
Then POST to `http://localhost:8000/ask` with JSON:
```json
{"query": "How does the system answer questions?"}
```

## Run tests
```bash
pip install -r requirements.txt
pytest -q
```

## Evaluation / Notes for AI judge
- The agent exposes a deterministic `RAGAgent.answer(query)` method. Tests validate:
  - retrieval returns relevant docs
  - agent composes an answer using retrieved context
  - tools can be invoked by the agent when plan requires computation
- To extend: plug real LLM generation (OpenAI, local LLM) into `rag_agent.py::generate_answer`.
