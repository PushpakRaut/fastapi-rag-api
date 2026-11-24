# FastAPI Docs Q&A (RAG) API

Upload PDFs and ask questions in plain English. Answers come only from your documents and cite the source pages.

**Live API docs:** https://pushpak-fastapi-rag.onrender.com/docs (free hosting: the first request after a quiet period takes about a minute while the server wakes up)

## What it shows

- **Hybrid retrieval:** BM25 keyword search and embedding similarity, merged with reciprocal rank fusion. Exact terms (product codes, names) and paraphrased questions both find the right passage. Without an embedding model it falls back to keyword search.
- **Grounded answers with citations:** the model is told to use only the numbered sources and cite them like `[1]`; the response lists file, page and excerpt for each source.
- **Any OpenAI-compatible model:** Gemini or OpenAI in the cloud, or a private local model through Ollama so documents never leave your machine.
- **Safe to expose publicly:** each visitor's documents are private (HttpOnly cookie session), with upload size, page, document and session caps, per-IP rate limits, and idle cleanup after 30 minutes.
- **Tested:** chunking, BM25, rank fusion, hybrid search, privacy between visitors, validation and rate limits (`pytest`).

## API

| Method | Path | What it does |
|---|---|---|
| POST | `/documents` | Upload a PDF (multipart `file`) |
| GET | `/documents` | List your documents |
| DELETE | `/documents/{name}` | Remove one |
| POST | `/ask` | `{"question": "..."}` → answer, sources, search mode |
| GET | `/health` | Health check |

## Run locally

```bash
uv sync
uv run uvicorn app.main:app --reload   # uses Ollama at localhost:11434 by default
uv run pytest
```

Configure with environment variables: `AI_BASE_URL`, `AI_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL` (empty = keyword only), `EMBED_DIMENSIONS`. See `app/config.py` for limits.

## Stack

Python 3.13, FastAPI, Pydantic, pypdf, httpx, pytest, Docker, Render.
