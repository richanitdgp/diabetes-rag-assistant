# app

FastAPI application: routes and RAG chain logic for serving the diabetes assistant.

- `retrieval.py` — embeds a query with Gemini (`task_type="RETRIEVAL_QUERY"`)
  and runs top-k similarity search against the local Chroma index built by
  `ingestion/build_index.py`. Raises if `data/chroma/` doesn't exist yet.
- `generation.py` — calls Gemini (`gemini-3.5-flash-lite`) with a strict
  system prompt: answer only from the retrieved passages, cite `[Source
  Title — Section Title]` inline for every claim, and refuse rather than
  guess when the context doesn't support a confident answer. This is the v1
  baseline to evaluate retrieval quality against (see `eval/`) — not yet
  optimized.
- `main.py` — `POST /ask` wires the two together: `{"question": str, "top_k":
  int}` in, `{"answer": str, "sources": [...]}` out. `GET /health` is a
  liveness check.

Requires `data/chroma/` to exist (`python -m ingestion.build_index`) and
`GOOGLE_API_KEY`/`GEMINI_API_KEY` set to answer questions.
