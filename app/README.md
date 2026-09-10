# app

FastAPI application: routes and RAG chain logic for serving the diabetes assistant.

- `retrieval.py` — embeds a query with Gemini (`task_type="RETRIEVAL_QUERY"`)
  and runs top-k similarity search via a MongoDB Atlas `$vectorSearch`
  aggregation against the collection `ingestion/build_index.py` populates.
  Raises if `MONGODB_URI` isn't set.
- `generation.py` — calls Gemini (`gemini-3.5-flash-lite`) with a strict
  system prompt: answer only from the retrieved passages, cite `[Source
  Title — Section Title]` inline for every claim, and refuse rather than
  guess when the context doesn't support a confident answer. This is the v1
  baseline to evaluate retrieval quality against (see `eval/`) — not yet
  optimized.
- `main.py` — `POST /ask` wires the two together: `{"question": str, "top_k":
  int}` in, `{"answer": str, "sources": [...]}` out. `GET /health` is a
  liveness check that does *not* verify the Atlas connection — a healthy
  process can still fail `/ask` if `MONGODB_URI` is wrong or the vector
  index isn't built yet.

Requires the Atlas collection to already be populated
(`python -m ingestion.build_index`, run separately — see `ingestion/README.md`)
and `GOOGLE_API_KEY`/`GEMINI_API_KEY` + `MONGODB_URI` set to answer questions.
